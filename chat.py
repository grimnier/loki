# chat.py -- terminal front end for the R1 geometry-memory runtime (server_core_async --chat). Standard library only.
#   Full-screen layout: the conversation fills from the bottom, the thinking hood sits above the input box (a pulled-down
#   shade showing the last 3 lines of thought; Tab lifts it), and the input box grows as you type. Enter sends, Ctrl-J
#   breaks a line, Ctrl-C stops the current answer (or quits when idle). Piped stdin falls back to a plain line mode.
#   chat.py [--replay [tok/s]] [--demo "first message"] [--keys "scripted keys"] [--frames FILE] [--size WxH]
import sys, os, json, time, queue, threading, subprocess, shutil, textwrap, datetime, re

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(HERE, "chat_settings.json")
DEFAULTS = {
    "think": 0,       # think tokens per turn before the thought is force-closed; 0 = unlimited (thoughts land complete; context is the only limit)
    "answer": 0,      # answer tokens per turn; 0 = unlimited (end-of-sequence or context decide)
    "bars": 3,        # lines of thought under the hood when it is down
    "lift": 12,       # lines when the hood is lifted (Tab)
    "stats": True,    # dim tok/s line after each answer
    "ctx": 1024,      # KV cache positions; with --mla (compressed latent cache) each costs ~0.13 MB VRAM, so 1024 keeps 14-15 layers resident (restart to change)
    "memory": "shared",   # shared: the model's memory grows across every conversation | sandbox: this conversation only, nothing loaded, nothing kept
    "temp": 0.0,      # sampling temperature at every head decision; 0 = greedy (the exact walk the memory stores and replays). >0 samples from the top_k / top_p set
    "top_k": 40,      # candidates considered when temp > 0 (0 = the whole vocabulary)
    "top_p": 0.95,    # nucleus cut inside those candidates when temp > 0
    "seed": 0,        # RNG seed for sampling; 0 = the engine's fixed default seed
    "stop": [],       # stop sequences: the answer ends (and is cut) at the first occurrence of any of these strings; the thought is not scanned
    # the fused engine (results/server_core/WRITEUP_2026-10-07.md, Oct 8 11:20): every layer as direct launches (--fused 1), next-layer experts read ahead on the
    # speculative route (--spec-read 1 with 12 prefetch slots, top-0 = no router prediction), drafter off; geo-sim 0.999 / vote 5 / text-cap 3 / text-min 3 are the engine's own defaults
    "engine_common": ["--draft-lookup", "8", "--draft-text", "reason_all.txt", "--experts", "2", "--share-experts", "--threads", "8", "--pinned-gb", "7.8", "--vram-keep-mb", "150", "--prefetch", "12", "--prefetch-top", "0", "--spec-read", "1", "--loader-inflight", "2", "--mla", "--fused", "1", "--prefill-chunk", "512"],
    # shared memory only: the accumulated geometry + route memory, the corpus of past answers, the state dump for drafter training
    "engine_shared": ["--route-memory", "ptx_trees/route_memory.bin", "--geo-memory", "ptx_trees/geo_memory.bin", "--draft-corpus", "ptx_trees/r1_draft_corpus.bin", "--dump-states", "results/sky_pull/states_dump.bin"],
}
SANDBOX_DIR = os.path.join(HERE, "results", "chat", "sandbox")

def engine_args(cfg):   # the engine flag list for the chosen memory mode; sandbox gets fresh scratch memory files that are deleted at exit
    if cfg.get("engine"): return list(cfg["engine"]), []            # an older settings file with one explicit list: use it as-is
    args = list(cfg["engine_common"])
    try:   # sampling dials -> engine flags (temp 0 passes nothing: the engine stays greedy)
        temp = float(cfg.get("temp", 0) or 0)
        if temp > 0:
            args += ["--temp", str(temp), "--top-k", str(int(cfg.get("top_k", 40) or 0)), "--top-p", str(float(cfg.get("top_p", 0.95) or 1.0))]
            if int(cfg.get("seed", 0) or 0): args += ["--seed", str(int(cfg["seed"]))]
    except (TypeError, ValueError): pass
    if "--pinned-gb" in args:   # the engine pins this much host RAM; cap it to what is free right now (minus ~2.3 GB for its own buffers) so a busy desktop does not fail the launch
        try:
            import ctypes
            class _MS(ctypes.Structure): _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS(); ms.dwLength = ctypes.sizeof(_MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            i = args.index("--pinned-gb"); want = float(args[i + 1]); fit = max(1.0, round((ms.ullAvailPhys / 1e6 - 3500) / 1000, 1))   # beyond the pinned budget the engine pins ~1.5 GB more (staging, slots, CUDA context) and the OS keeps a reserve; measured: 5.9 GB launched at 7.9 GB free, 6.7 GB died at 8.7 GB free (host RAM exhausted -> CUDA 'out of memory' at startup), so leave 3 GB
            if fit < want: args[i + 1] = str(fit)
        except Exception: pass
    if cfg.get("memory") == "sandbox":
        os.makedirs(SANDBOX_DIR, exist_ok=True); tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + str(os.getpid())
        geo = os.path.join(SANDBOX_DIR, f"geo_{tag}.bin"); route = os.path.join(SANDBOX_DIR, f"route_{tag}.bin")
        return args + ["--geo-memory", geo, "--route-memory", route], [geo, geo + ".out", geo + ".tmp", route]
    return args + list(cfg["engine_shared"]), []
U, A, EOS = "<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"
DIM, RESET = "\x1b[2m", "\x1b[0m"
THINK_CLOSE, ENDTOK, NL = 128799, 1, 201

def load_settings():
    cfg = dict(DEFAULTS)
    try: cfg.update(json.load(open(SETTINGS, encoding="utf-8")))
    except Exception: pass
    return cfg

def save_settings(cfg):
    try: json.dump(cfg, open(SETTINGS, "w", encoding="utf-8"), indent=1)
    except Exception: pass

def console_setup():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32; k.SetConsoleOutputCP(65001); k.SetConsoleCP(65001)
            h = k.GetStdHandle(-11); m = ctypes.c_uint32()
            if k.GetConsoleMode(h, ctypes.byref(m)): k.SetConsoleMode(h, m.value | 0x0004)
        except Exception: pass

def out(s=""):
    sys.stdout.write(s); sys.stdout.flush()

def fmt_t(sec):
    sec = int(round(sec)); return f"{sec // 60}m {sec % 60:02d}s" if sec >= 60 else f"{sec}s"

def esc(s): return s.replace("\\", "\\\\").replace("\r", "").replace("\n", "\\n")

def unesc(s):
    o, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s): i += 1; o.append("\n" if s[i] == "n" else s[i])
        else: o.append(s[i])
        i += 1
    return "".join(o)

# ---------------------------------------------------------------- engine (the runtime over its stdin/stdout protocol)
class Engine:
    def __init__(self, cfg):
        bat = os.path.join(HERE, "run_server_core_fused.bat"); flags, self.scratch = engine_args(cfg)   # the fused-layer build (src/server_core_fused.exe); run_server_core_async.bat is the ggml-graph build
        args = ["cmd" if os.name == "nt" else "cmd.exe", "/c", bat, "--chat", "--ctx", str(int(cfg["ctx"]))] + flags
        elog = open(os.path.join(HERE, "results", "server_core", "chat_engine.log"), "w", encoding="utf-8", errors="replace")   # the engine's pass-by-pass log (the UI points here when the engine stops)
        self.p = subprocess.Popen(args, cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=elog)
        self.q = queue.Queue(); self.ready = False; self.ctx = int(cfg["ctx"]); self.pos = 0
        threading.Thread(target=self._pump, daemon=True).start()
    def cleanup(self):   # sandbox: the scratch memory files go with the conversation
        for f in self.scratch:
            try: os.remove(f)
            except OSError: pass
    @staticmethod
    def sweep_sandbox():   # scratch files left by an earlier crash
        try:
            for f in os.listdir(SANDBOX_DIR):
                try: os.remove(os.path.join(SANDBOX_DIR, f))
                except OSError: pass
        except OSError: pass
    def _pump(self):
        for raw in iter(self.p.stdout.readline, b""): self.q.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
        self.q.put(None)
    def send(self, line):
        try: self.p.stdin.write((line + "\n").encode("utf-8")); self.p.stdin.flush()
        except Exception: pass
    def get(self, timeout=0.25):
        try: return self.q.get(timeout=timeout)
        except queue.Empty: return ""
    def wait_for(self, *heads):
        while True:
            m = self.get(0.5)
            if m is None: return None
            if m and m.split(" ", 1)[0] in heads: return m
    def alive(self): return self.p.poll() is None

REPLAY = [   # --replay: turns captured from the real runtime on 2026-10-07 (think closed at a 20-token budget; the cut-off answers are finished here for the demo)
    ("What color is the sky?",
     "Okay, the user is asking, \"What color is the sky?\" Let me start by recalling the basic answer. Normally, the sky is blue on a clear day. But it changes with time of day and weather, so I should mention that too. Rayleigh scattering is the mechanism, and I should keep the explanation short and plain.",
     "The color of the sky can vary depending on the time of day and atmospheric conditions. Typically, during the day it appears blue because air scatters blue light more than red light. At sunrise and sunset it shifts to orange and red, and on overcast days it looks grey."),
    ("And at sunset?",
     "Alright, the user just asked, \"And at sunset?\" following my previous answer about the sky's color. They want the sunset case specifically, so I should explain the longer path through the atmosphere and why the blue is gone by the time the light arrives.",
     "At sunset, the sky often displays vibrant hues of red, orange, pink, and purple. This happens because sunlight travels through more atmosphere near the horizon, so most of the blue light is scattered away before it reaches you and the longer red wavelengths remain."),
    ("Why do we have eyebrows?",
     "Okay, so the user is asking, \"Why do we have eyebrows?\" Hmm, I need to cover the practical functions first, then the social side, and keep it short. Protection from sweat, expression, recognition. Three points should do it.",
     "Eyebrows serve several important functions:\n\n1. Protection: they help shield the eyes from sweat, rain, and debris, channeling moisture to the sides of the face.\n2. Communication: they are central to facial expressions, which is why they are so expressive in conversation.\n3. Recognition: studies show faces are harder to recognise without eyebrows than without eyes."),
]

class Replay:   # stands in for the engine: same protocol, canned turns, a chosen pace; nothing is loaded
    def __init__(self, cfg, rate):
        self.q = queue.Queue(); self.ready = False; self.ctx = int(cfg["ctx"]); self.pos = 0; self.rate = max(0.5, rate); self.n = 0; self.stop = False; self.p = None; self.scratch = []
        self.split = lambda t: re.findall(r"[ \t]*\S+|\n", t)
        threading.Thread(target=lambda: (time.sleep(1.5), self.q.put(f"READY {self.ctx}")), daemon=True).start()
    def alive(self): return self.p is None
    def send(self, line):
        if line == "X": self.stop = True
        elif line == "R": self.pos = 0; self.q.put("RESET")
        elif line == "Q": self.p = 0; self.q.put(None)
        elif line.startswith("U "): threading.Thread(target=self._turn, args=(line,), daemon=True).start()
    def _turn(self, line):
        _, think_b, answer_b, text = line.split(" ", 3); think_b, answer_b = int(think_b), int(answer_b); text = unesc(text)
        msg = text.rsplit(U, 1)[1].split(A, 1)[0].lower()
        q, think, answer = next((r for r in REPLAY if r[0].lower() in msg or msg in r[0].lower()), REPLAY[self.n % len(REPLAY)]); self.n += 1
        self.stop = False; t0 = time.time(); tn = an = 0; self.pos += len(self.split(text)) + 2
        def emit(kind, piece, tid=7):
            time.sleep(1.0 / self.rate); self.q.put(f"{kind} {tid} {esc(piece)}"); self.pos += 1
        for piece in self.split(think):
            if self.stop or (think_b > 0 and tn >= think_b) or self.pos >= self.ctx - 2: break
            emit("T", piece); tn += 1
        if self.stop: self.q.put(f"END stop {tn} {an} {time.time() - t0:.2f} {self.pos} {self.ctx}"); return
        if tn >= think_b:
            for piece in self.split("\n\nOkay, I have thought about this enough. Let me answer now.\n"): emit("F", piece)
        emit("F" if tn >= think_b else "T", "</think>", THINK_CLOSE); emit("T", "\n", NL); emit("T", "\n", NL)
        for piece in self.split(answer):
            if self.stop or (answer_b > 0 and an >= answer_b) or self.pos >= self.ctx - 2: break
            emit("T", piece); an += 1
        reason = "stop" if self.stop else ("cap" if (answer_b > 0 and an >= answer_b) else ("full" if self.pos >= self.ctx - 2 else "eos"))
        if reason == "eos": emit("T", "", ENDTOK)
        self.q.put(f"END {reason} {tn} {an} {time.time() - t0:.2f} {self.pos} {self.ctx}")
    def get(self, timeout=0.25):
        try: return self.q.get(timeout=timeout)
        except queue.Empty: return ""
    wait_for = Engine.wait_for; cleanup = Engine.cleanup

# ---------------------------------------------------------------- keyboard sources
class Quit(Exception): pass

class ConsoleKeys:   # raw keys from the terminal: msvcrt on Windows, termios on POSIX
    WIN = {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT", "G": "HOME", "O": "END", "S": "DEL", "I": "PGUP", "Q": "PGDN"}
    ESC = {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT", "[H": "HOME", "[F": "END", "[1~": "HOME", "[4~": "END", "[3~": "DEL", "[5~": "PGUP", "[6~": "PGDN", "OH": "HOME", "OF": "END"}
    def __init__(self):
        self.win = os.name == "nt"
        if not self.win:
            import termios, tty
            self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); tty.setcbreak(self.fd)
    def close(self):
        if not self.win:
            import termios; termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def poll(self, timeout):
        keys = []
        if self.win:
            import msvcrt
            t_end = time.time() + timeout
            while True:
                while msvcrt.kbhit():
                    c = msvcrt.getwch()
                    if c in ("\x00", "\xe0"): keys.append(self.WIN.get(msvcrt.getwch(), ""))
                    else: keys.append(self.plain(c))
                if keys or time.time() >= t_end: return self.batch(keys)
                time.sleep(0.01)
        import select
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r: return keys
        data = os.read(self.fd, 4096).decode("utf-8", "replace"); i = 0
        while i < len(data):
            c = data[i]
            if c == "\x1b" and i + 1 < len(data):
                for n in (3, 2):
                    if data[i + 1:i + 1 + n] in self.ESC: keys.append(self.ESC[data[i + 1:i + 1 + n]]); i += 1 + n; break
                else: keys.append("ESC"); i += 1
                continue
            keys.append(self.plain(c)); i += 1
        return self.batch(keys)
    @staticmethod
    def batch(keys):   # a typed Enter arrives alone; an Enter with more keys behind it is a pasted line break
        keys = [k for k in keys if k]
        return [("NL" if k == "ENTER" and i < len(keys) - 1 else k) for i, k in enumerate(keys)]
    @staticmethod
    def plain(c):
        return {"\r": "ENTER", "\n": "NL", "\t": "TAB", "\x08": "BS", "\x7f": "BS", "\x1b": "ESC", "\x03": "CTRL_C", "\x04": "CTRL_C"}.get(c, c if c >= " " else "")

class ScriptedKeys:   # --keys: literal text plus <ENTER> <TAB> <UP> <DOWN> <PGUP> <BS> <CTRLC> <WAIT:s> <QUIT>, one key per poll
    def __init__(self, script):
        self.items = [m if m.startswith("<") else m for m in re.findall(r"<[^>]+>|.", script)]
        self.wait_until = 0
    def close(self): pass
    def poll(self, timeout):
        if time.time() < self.wait_until: time.sleep(min(timeout, self.wait_until - time.time())); return []
        if not self.items: time.sleep(timeout); return []
        k = self.items.pop(0)
        if k.startswith("<"):
            name = k[1:-1]
            if name.startswith("WAIT:"): self.wait_until = time.time() + float(name[5:]); return []
            return [{"CTRLC": "CTRL_C"}.get(name, name)]
        return [k]

# ---------------------------------------------------------------- the screen
class Screen:
    def __init__(self, frames=None, size=None):
        self.frames = open(frames, "w", encoding="utf-8") if frames else None; self.size_fixed = size; self.n = 0; self.live = not frames
        if self.live: out("\x1b[?1049h\x1b[?25l\x1b[H\x1b[2J")
    def size(self):
        if self.size_fixed: return self.size_fixed
        s = shutil.get_terminal_size((100, 30)); return max(40, s.columns), max(12, s.lines)
    def draw(self, rows, cursor):
        w, h = self.size()
        if (rows, cursor, w, h) == getattr(self, "last", None): return
        self.last = (list(rows), cursor, w, h)
        if self.live:
            s = "\x1b[?25l"
            for i, r in enumerate(rows[:h]): s += f"\x1b[{i + 1};1H{r}\x1b[K"
            s += f"\x1b[{cursor[1]};{cursor[0]}H\x1b[?25h"; out(s)
        else:
            self.n += 1; self.frames.write(f"---- frame {self.n} ({w}x{h}) cursor {cursor}\n" + "\n".join(re.sub(r"\x1b\[[0-9;]*m", "", r) for r in rows) + "\n"); self.frames.flush()
    def close(self):
        if self.live: out("\x1b[?25h\x1b[?1049l")
        else: self.frames.close()

def wrap(text, w):
    lines = []
    for para in text.split("\n"): lines.extend(textwrap.wrap(para, w, drop_whitespace=True) or [""])
    return lines

# ---------------------------------------------------------------- the full-screen chat
class TUI:
    def __init__(self, cfg, eng, keys, screen):
        self.cfg, self.eng, self.keys, self.screen = cfg, eng, keys, screen
        self.log = []            # {"kind": user|answer|dim, "text": str}
        self.inp = ""; self.caret = 0; self.scroll = 0; self.lifted = False
        self.turn = None; self.history = []; self.dead = False; self.last_tick = 0
        os.makedirs(os.path.join(HERE, "results", "chat"), exist_ok=True)
        self.logfile = os.path.join(HERE, "results", "chat", datetime.date.today().isoformat() + ".md")

    # ---- input handling
    def handle(self, k):
        if k == "CTRL_C":
            if self.turn and not self.turn["stopped"]: self.eng.send("X"); self.turn["stopped"] = True
            elif not self.turn: raise Quit()
        elif k == "ENTER":
            if not self.turn: self.submit()
        elif k == "NL": self.insert("\n")
        elif k == "BS":
            if self.caret: self.inp = self.inp[:self.caret - 1] + self.inp[self.caret:]; self.caret -= 1
        elif k == "DEL": self.inp = self.inp[:self.caret] + self.inp[self.caret + 1:]
        elif k == "LEFT": self.caret = max(0, self.caret - 1)
        elif k == "RIGHT": self.caret = min(len(self.inp), self.caret + 1)
        elif k == "HOME": self.caret = 0
        elif k == "END": self.caret = len(self.inp)
        elif k in ("UP", "PGUP"): self.scroll += 1 if k == "UP" else 10
        elif k in ("DOWN", "PGDN"): self.scroll = max(0, self.scroll - (1 if k == "DOWN" else 10))
        elif k == "TAB": self.lifted = not self.lifted
        elif k == "ESC": self.inp = ""; self.caret = 0
        elif k == "QUIT": raise Quit()
        elif len(k) == 1: self.insert(k)
        else: self.insert(k)   # pasted text arrives as plain characters
    def insert(self, s):
        self.inp = self.inp[:self.caret] + s + self.inp[self.caret:]; self.caret += len(s)
    def note(self, text): self.log.append({"kind": "dim", "text": text})
    def submit(self):
        text = self.inp.strip(); self.inp = ""; self.caret = 0; self.scroll = 0
        if not text: return
        if text.startswith("/"): self.command(text); return
        if self.dead: self.note("engine stopped · see results/server_core/chat_engine.log"); return
        self.log.append({"kind": "user", "text": text})
        self.turn = {"msg": text, "think": "", "ans": "", "t0": time.time(), "in_think": True, "stopped": False, "stage": "ready?", "attempt": 0, "think_n": 0, "answer_n": 0}
        if self.eng.ready: self.send_turn()
    def send_turn(self):
        t = self.turn; think, answer = int(self.cfg["think"]), int(self.cfg["answer"])
        if answer == 0: answer = 1 << 20                                    # unlimited: the engine build in use clamps 0 up to 1, so send "huge" instead
        est = len(t["msg"]) // 3 + 8; need = est + min(think or 160, 160) + min(answer, 120)
        if t["attempt"] == 0 and self.eng.ctx - self.eng.pos >= need:
            text = f"{U}{t['msg']}{A}<think>\n"
        elif t["attempt"] <= 1:                                      # will not fit after what is there: start over, carrying the last exchange when it fits
            if t["stage"] != "reset": self.eng.send("R"); t["stage"] = "reset"; return
            carry = [(u, a) for u, a in self.history[-1:] if a and t["attempt"] == 0 and len(u + a) // 3 + 8 <= self.eng.ctx - need]
            text = "".join(f"{U}{u}{A}{a}{EOS}" for u, a in carry) + f"{U}{t['msg']}{A}<think>\n"; self.history = []
        else:
            self.note("message is too long for the context"); self.turn = None; return
        self.eng.send(f"U {think} {answer} {esc(text)}"); t["stage"] = "sent"

    # ---- engine messages
    def pump(self):
        while True:
            m = self.eng.get(0)
            if m is None: self.dead = True; self.note("engine stopped · see results/server_core/chat_engine.log"); self.turn = None; return
            if not m: return
            head, _, rest = m.partition(" ")
            if head == "READY":
                self.eng.ready = True; self.eng.ctx = int(rest)
                if self.turn and self.turn["stage"] == "ready?": self.send_turn()
            elif head == "RESET":
                self.eng.pos = 0
                if self.turn and self.turn["stage"] == "reset": self.send_turn()
            elif head == "FULL":
                if self.turn: self.turn["attempt"] += 1; self.send_turn()
            elif head in ("T", "F") and self.turn:
                tid, _, piece = rest.partition(" "); tid = int(tid); piece = unesc(piece); t = self.turn
                if t["in_think"]:
                    if tid == THINK_CLOSE: self.close_hood()
                    elif tid != ENDTOK: t["think"] += piece
                elif tid != ENDTOK:
                    t["ans"] += piece; self.check_stop(t)
                    if not self.log or self.log[-1]["kind"] != "answer": self.log.append({"kind": "answer", "text": ""})
                    self.log[-1]["text"] = t["ans"].lstrip("\n")
            elif head == "END" and self.turn: self.finish(rest.split())
    def check_stop(self, t):   # stop sequences (settings "stop"): the answer is cut at the first occurrence and the engine is asked to end the turn
        if t.get("stopped"): return
        for s in (self.cfg.get("stop") or []):
            if s and s in t["ans"]: t["ans"] = t["ans"][:t["ans"].index(s)]; t["stopped"] = True; self.eng.send("X"); break
    def close_hood(self):
        t = self.turn
        for n in (2, 1, 0):                                          # the shade comes down over the thought
            t["hood_h"] = n; self.render(); time.sleep(0.04)
        t["in_think"] = False; t.pop("hood_h", None); self.note(f"thought for {fmt_t(time.time() - t['t0'])}")
    def finish(self, end):
        t = self.turn; reason, think_n, answer_n, secs, pos, ctx = end[0], int(end[1]), int(end[2]), float(end[3]), int(end[4]), int(end[5])
        self.eng.pos, self.eng.ctx = pos, ctx
        if t["in_think"]: self.close_hood()
        answer = t["ans"].strip(); thought = t["think"].strip(); notes = []
        if reason == "stop": notes.append("stopped")
        elif reason == "full": notes.append("context full" if answer else ("ran out of room while thinking · /think N forces an answer, /ctx higher gives it room" if not int(self.cfg["think"]) else "ran out of room while thinking · /think lower or /ctx higher"))
        elif not answer: notes.append("no answer · /answer higher")
        if self.cfg["stats"] and (think_n + answer_n) and secs > 0: notes.append(f"{(think_n + answer_n) / secs:.1f} tok/s · {think_n + answer_n} tokens · {fmt_t(secs)}")
        if notes: self.note(" · ".join(notes))
        self.history.append((t["msg"], answer)); self.turn = None
        try:
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(f"\n### {datetime.datetime.now():%H:%M}\n\n**you:** {t['msg']}\n\n> {thought.replace(chr(10), chr(10) + '> ')}\n\n**r1:** {answer}\n")
        except Exception: pass

    # ---- commands
    def command(self, line):
        parts = line.split(); c = parts[0].lower(); arg = parts[1] if len(parts) > 1 else None
        if c in ("/quit", "/q", "/exit"): raise Quit()
        if c == "/new":
            if self.turn: self.note("wait for the answer, or Ctrl-C"); return
            if self.eng.ready: self.eng.send("R")
            self.log = []; self.history = []; self.scroll = 0; return
        if c in ("/think", "/answer", "/bars", "/lift"):
            key = c[1:]
            if arg is None or not arg.isdigit(): self.note(f"{key} {self.cfg[key]}" + (" (unlimited)" if key in ("think", "answer") and not self.cfg[key] else "")); return
            self.cfg[key] = max(1 if key == "lift" else 0, int(arg)); save_settings(self.cfg); self.note(f"{key} {self.cfg[key]}" + (" (unlimited)" if key in ("think", "answer") and not self.cfg[key] else "")); return
        if c == "/stats":
            self.cfg["stats"] = not self.cfg["stats"]; save_settings(self.cfg); self.note("stats " + ("on" if self.cfg["stats"] else "off")); return
        if c in ("/sandbox", "/memory"):
            if c == "/memory" and arg is None: self.note("memory " + self.cfg.get("memory", "shared")); return
            if self.turn: self.note("wait for the answer, or Ctrl-C"); return
            want = {"on": "sandbox", "off": "shared", "sandbox": "sandbox", "shared": "shared"}.get(arg, None) if arg else ("shared" if self.cfg.get("memory") == "sandbox" else "sandbox")
            if want is None: self.note("/sandbox [on|off]"); return
            self.cfg["memory"] = want; save_settings(self.cfg); self.restart_engine()
            self.note("memory sandbox · this conversation only, nothing kept" if want == "sandbox" else "memory shared · grows across conversations"); return
        if c == "/ctx":
            if arg and arg.isdigit(): self.cfg["ctx"] = max(64, int(arg)); save_settings(self.cfg); self.note(f"ctx {self.cfg['ctx']} from the next start"); return
            self.note(f"{self.eng.pos} of {self.eng.ctx} used"); return
        if c == "/help":
            self.note("/new  fresh conversation     /think N  think tokens     /answer N  answer tokens     /bars N  lines under the hood\n"
                      "/lift N  lines when lifted   /stats  speed line         /ctx [N]  context          /quit\n"
                      "/sandbox [on|off]  memory for this conversation only (nothing loaded from or kept for other chats); /memory shows the mode\n"
                      "Tab  lift / lower the hood   Up Down PgUp PgDn  scroll  Ctrl-J  new line in the box  Ctrl-C  stop the answer"); return
        self.note("no such command · /help")
    def restart_engine(self):   # the memory mode is fixed at engine start: stop this engine (it saves what it should), start a fresh one, fresh conversation
        self.quit(quiet=True); self.log = []; self.history = []; self.scroll = 0; self.dead = False
        self.eng = Replay(self.cfg, self.eng.rate) if isinstance(self.eng, Replay) else Engine(self.cfg)

    # ---- drawing
    def log_lines(self, w):
        lines = []
        for e in self.log:
            if e["kind"] == "user": lines += [""] + [(DIM + "› " + RESET + l if i == 0 else "  " + l) for i, l in enumerate(wrap(e["text"], w - 4))]
            elif e["kind"] == "answer": lines += ["  " + l for l in wrap(e["text"], w - 4)]
            else: lines += [DIM + "  " + l + RESET for l in wrap(e["text"], w - 4)]
        return lines
    def hood_rows(self, w):
        t = self.turn
        if not t or not t["in_think"]: return []
        n = t.get("hood_h", int(self.cfg["lift"]) if self.lifted else int(self.cfg["bars"]))
        lines = wrap(t["think"], w - 6)[-n:] if n else []
        lines += [""] * (n - len(lines))
        head = f" thinking · {fmt_t(time.time() - t['t0'])}" + (" · stopping" if t["stopped"] else "") + " "
        band = "▓" * 3 + head + "▓" * max(0, w - 3 - len(head) - 1)
        return [DIM + band + RESET, DIM + "░" * (w - 1) + RESET] + [DIM + "   " + l + RESET for l in lines]
    def box_rows(self, w):   # the input box: word-wrapped to the box, one row per line, grows as you type; returns rows and the caret's (row, col)
        inner = w - 6; segs = []; start = 0; text = self.inp
        while True:
            nl = text.find("\n", start); end = nl if nl >= 0 else len(text)
            while end - start > inner:                                   # wrap at the last space that fits, else hard
                cut = text.rfind(" ", start, start + inner + 1)
                cut = cut + 1 if cut > start else start + inner
                segs.append((start, text[start:cut])); start = cut
            segs.append((start, text[start:end]))
            if nl < 0: break
            start = nl + 1
        pos = (0, 0)
        for r, (s0, s) in enumerate(segs):                              # a caret on a wrap boundary sits at the start of the next row
            if s0 <= self.caret <= s0 + len(s): pos = (r, min(self.caret - s0, inner))
        segs = [(s0, s[:-1] if s.endswith(" ") and len(s) > inner else s) for s0, s in segs]   # the wrap space stays in the text, not in the box
        top = DIM + "╭" + "─" * (w - 3) + "╮" + RESET; bot = DIM + "╰" + "─" * (w - 3) + "╯" + RESET
        rows = [top] + [DIM + "│" + RESET + (" › " if k == 0 else "   ") + s + " " * (inner - len(s)) + " " + DIM + "│" + RESET for k, (_, s) in enumerate(segs)] + [bot]
        return rows, pos
    def render(self):
        w, h = self.screen.size(); box, (crow, ccol) = self.box_rows(w); hood = self.hood_rows(w)
        log_h = max(1, h - len(box) - len(hood)); lines = self.log_lines(w)
        self.scroll = min(self.scroll, max(0, len(lines) - log_h))
        end = len(lines) - self.scroll; vis = lines[max(0, end - log_h):end]
        rows = [""] * (log_h - len(vis)) + vis + hood + box
        self.screen.draw(rows, (ccol + 5, h - len(box) + 2 + crow))

    def run(self, demo=None):
        try:
            if demo: self.inp = demo; self.submit()
            while True:
                try:
                    for k in self.keys.poll(0.03): self.handle(k)
                    self.pump(); self.render()
                except KeyboardInterrupt: self.handle("CTRL_C")
        except Quit: pass
        finally:
            self.keys.close(); self.screen.close(); self.quit()
    def quit(self, quiet=False):
        if self.eng.alive():
            self.eng.send("Q"); t0 = time.time(); shown = False
            while self.eng.alive() and time.time() - t0 < 300:
                time.sleep(0.2)
                if not shown and not quiet and time.time() - t0 > 2.0: out(DIM + "saving memory" + RESET + "\n"); shown = True
            if self.eng.alive() and getattr(self.eng, "p", None): self.eng.p.kill()
        self.eng.cleanup()

# ---------------------------------------------------------------- plain line mode (piped stdin)
class LineChat:
    def __init__(self, cfg, eng):
        self.cfg, self.eng, self.history = cfg, eng, []
    def wait_ready(self):
        while not self.eng.ready:
            m = self.eng.get(0.5)
            if m is None: return False
            if m.startswith("READY"): self.eng.ready = True; self.eng.ctx = int(m.split()[1])
        return True
    def turn(self, msg):
        think, answer = int(self.cfg["think"]), int(self.cfg["answer"]) or (1 << 20); need = len(msg) // 3 + 8 + min(think or 160, 160) + min(answer, 120)
        if self.eng.ctx - self.eng.pos < need: self.eng.send("R"); self.eng.wait_for("RESET"); self.eng.pos = 0
        self.eng.send(f"U {think} {answer} {esc(U + msg + A + '<think>' + chr(10))}"); in_think = True; ans = []; stopped = False
        while True:
            m = self.eng.get(1.0)
            if m is None: return False
            if not m: continue
            head, _, rest = m.partition(" ")
            if head in ("T", "F"):
                tid, _, piece = rest.partition(" "); tid = int(tid); piece = unesc(piece)
                if in_think:
                    if tid == THINK_CLOSE: in_think = False; out(DIM + "\n[answer]" + RESET + "\n")
                    else: out(DIM + piece + RESET)
                elif tid != ENDTOK and not stopped:   # stop sequences (settings "stop"): print up to the first occurrence, then ask the engine to end the turn
                    full = "".join(ans) + piece; cut = -1
                    for s in (self.cfg.get("stop") or []):
                        if s and s in full: cut = full.index(s); break
                    if cut >= 0: piece = piece[:max(0, cut - len("".join(ans)))]; stopped = True; self.eng.send("X")
                    ans.append(piece); out(piece)
            elif head == "FULL": out(DIM + "message is too long for the context" + RESET + "\n"); return True
            elif head == "END":
                e = rest.split(); self.eng.pos, self.eng.ctx = int(e[4]), int(e[5]); out(f"\n{DIM}{e[0]} · {e[1]} think · {e[2]} answer · {e[3]}s{RESET}\n"); return True
    def run(self, demo=None):
        if not self.wait_ready(): return
        line = demo
        while True:
            if line is None:
                try: line = input("> ")
                except (EOFError, KeyboardInterrupt): break
            line = line.strip()
            if line in ("/quit", "/q", "/exit"): break
            if line and not line.startswith("/") and not self.turn(line): break
            line = None
        TUI.quit(self)

if __name__ == "__main__":
    console_setup(); argv = sys.argv[1:]
    def opt(name, default=None, conv=str):
        if name not in argv: return None
        i = argv.index(name); return conv(argv[i + 1]) if i + 1 < len(argv) and not argv[i + 1].startswith("--") else default
    replay = opt("--replay", 4.0, float); demo = opt("--demo", "What color is the sky?"); keys = opt("--keys", ""); frames = opt("--frames"); size = opt("--size")
    cfg = load_settings()
    if "--sandbox" in argv: cfg["memory"] = "sandbox"      # this run only; /sandbox inside the chat saves the choice
    if "--shared" in argv: cfg["memory"] = "shared"
    Engine.sweep_sandbox(); eng = Replay(cfg, replay) if replay else Engine(cfg)
    if keys is not None or frames:
        TUI(cfg, eng, ScriptedKeys(keys or ""), Screen(frames, tuple(int(v) for v in size.split("x")) if size else None)).run(demo)
    elif sys.stdin.isatty() and sys.stdout.isatty():
        TUI(cfg, eng, ConsoleKeys(), Screen()).run(demo)
    else:
        LineChat(cfg, eng).run(demo)
