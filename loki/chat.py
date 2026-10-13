# chat.py -- terminal front end for the R1 geometry-memory runtime (server_core_async --chat). Standard library only.
#   Full-screen layout: the conversation fills from the bottom, the thinking hood sits above the input box (a dim status
#   line and the last 4 lines of thought; Tab opens it over the screen with the thought scrollable), and the input box
#   grows as you type. Enter sends (or queues, while the model is answering), Ctrl-J breaks a line, Esc stops the turn
#   or clears the box, Ctrl-C stops the turn or quits on a second press. "/" opens the command palette. Text arrives at
#   a steady pace whatever the engine's bursts; the hood opens and closes on an easing curve; only changed rows are
#   repainted. Piped stdin falls back to a plain line mode.
#   chat.py [--replay [tok/s]] [--demo "first message"] [--keys "scripted keys"] [--frames FILE] [--size WxH] [--sandbox|--shared] [--ascii] [--no-mouse]
import sys, os, json, time, queue, threading, subprocess, shutil, datetime, re, collections

LOKI = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(LOKI)   # LOKI: this client; HERE: the repo, where the engine, settings and results live
sys.path.insert(0, LOKI)
import tools, term   # the tool belt (loki/tools): what the model can ask the client to do; term: the screen, keys, widths, glyphs
from term import DIM, RESET, G, out, dw, pad, clip, wrap, flow, ease_out, FPS, Screen, ConsoleKeys, ScriptedKeys, Quit
SETTINGS = os.path.join(HERE, "chat_settings.json")
CONVS = os.path.join(HERE, "results", "conversations")
DRAFT = os.path.join(HERE, "results", "chat", "draft.txt")
DEFAULTS = {
    "think": 0,       # think tokens per turn before the thought is force-closed; 0 = unlimited (thoughts land complete; context is the only limit)
    "answer": 0,      # answer tokens per turn; 0 = unlimited (end-of-sequence or context decide)
    "bars": 4,        # lines of thought under the hood (Tab opens it over the screen)
    "stats": True,    # dim tok/s line after each answer
    "mouse": True,    # the wheel scrolls the conversation (SGR mouse reporting; Windows Terminal through VT input)
    "bell": True,     # the terminal bell when an answer that took a while lands (you walked away); the window notifies when it is not focused
    "attach_cap": 6000,   # characters of an @file the model sees with the message
    "tools": "auto",  # offer the tool belt (tools/) to the model: true, false, or auto = only when ctx is 2048 or more (the list costs ~1100 tokens); /tools lists them
    "hops": 4,        # tool calls allowed in one turn before the model has to answer
    "tool_cap": 4000, # characters of a tool result the model sees (about a thousand tokens); tool_caps {"fetch": 8000} per tool
    "tool_policy": {},   # per-tool overrides of allow / ask / off
    "keys": {},       # API keys for tools that want one, e.g. {"BRAVE_API_KEY": "..."}; environment variables work too
    "presets": {"fast": {"ctx": 128, "think": 60, "answer": 80, "tools": False}, "deep": {"ctx": 2048, "think": 0, "answer": 0, "tools": "auto"}},   # /fast and /deep
    "ctx": 256,       # KV cache positions; ~5 MB VRAM each, so 384 pushes six more layers off the GPU and halves the speed (restart to change)
    "memory": "shared",   # shared: the model's memory grows across every conversation | sandbox: this conversation only, nothing loaded, nothing kept
    # the floor command from results/server_core/WRITEUP_2026-10-07.md (03:10): drafter off, prefetch off; geo-sim 0.99 / vote 5 / text-cap 3 / text-min 3 are the engine's own defaults
    "engine_common": ["--draft-lookup", "8", "--draft-text", "reason_all.txt", "--experts", "3", "--share-experts", "--threads", "8", "--pinned-gb", "7.4", "--vram-keep-mb", "300", "--prefetch", "0"],
    # shared memory only: the accumulated geometry + route memory, the corpus of past answers, the state dump for drafter training
    "engine_shared": ["--route-memory", "ptx_trees/route_memory.bin", "--geo-memory", "ptx_trees/geo_memory.bin", "--draft-corpus", "ptx_trees/r1_draft_corpus.bin", "--dump-states", "results/sky_pull/states_dump.bin"],
}
SANDBOX_DIR = os.path.join(HERE, "results", "chat", "sandbox")
ENGINE_LOG = os.path.join(HERE, "results", "server_core", "chat_engine.log")
HOOD_CLOSE, HOOD_OPEN = 0.26, 0.2   # seconds the hood takes to drop over a finished thought, and to open or close on Tab

def engine_args(cfg):   # the engine flag list for the chosen memory mode; sandbox gets fresh scratch memory files that are deleted at exit
    if cfg.get("engine"): return list(cfg["engine"]), []            # an older settings file with one explicit list: use it as-is
    args = list(cfg["engine_common"])
    if cfg.get("memory") == "sandbox":
        os.makedirs(SANDBOX_DIR, exist_ok=True); tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + str(os.getpid())
        geo = os.path.join(SANDBOX_DIR, f"geo_{tag}.bin"); route = os.path.join(SANDBOX_DIR, f"route_{tag}.bin")
        return args + ["--geo-memory", geo, "--route-memory", route], [geo, geo + ".out", geo + ".tmp", route]
    return args + list(cfg["engine_shared"]), []
U, A, EOS = "<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"
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

def fmt_t(sec):
    sec = int(round(sec)); return f"{sec // 60}m {sec % 60:02d}s" if sec >= 60 else f"{sec}s"

def esc(s): return s.replace("\\", "\\\\").replace("\r", "").replace("\n", "\\n")

def clipboard(text):   # whichever clipboard tool the machine has; True when one took the text
    for cmd in (["clip"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["pbcopy"]):
        if shutil.which(cmd[0]):
            try: subprocess.run(cmd, input=text.encode("utf-8"), check=True, timeout=5); return True
            except Exception: pass
    return False

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
        bat = os.path.join(HERE, cfg.get("engine_bat", "run_server_core_async.bat")); flags, self.scratch = engine_args(cfg)   # engine_bat in chat_settings.json picks the launcher (the laptop runs the fused build)
        args = ["cmd" if os.name == "nt" else "cmd.exe", "/c", bat, "--chat", "--ctx", str(int(cfg["ctx"]))] + flags
        self.p = subprocess.Popen(args, cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.q = queue.Queue(); self.ready = False; self.ctx = int(cfg["ctx"]); self.pos = 0
        self.stats = collections.deque(maxlen=400)   # the engine's stderr lines ([live], pass k:, [geometry], the end-of-turn summaries): the pace the interface shows
        threading.Thread(target=self._pump, daemon=True).start(); threading.Thread(target=self._stderr, daemon=True).start()
    def _stderr(self):   # stderr goes to results/server_core/chat_engine.log as before, and to the stats feed
        os.makedirs(os.path.dirname(ENGINE_LOG), exist_ok=True)
        try: log = open(ENGINE_LOG, "a", encoding="utf-8", errors="replace")
        except OSError: log = None
        for raw in iter(self.p.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n"); self.stats.append(line)
            if log:
                try: log.write(line + "\n"); log.flush()
                except OSError: pass
        if log: log.close()
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
    ("What time is it?",   # a tool call (synthetic): the answer ends in a call, the client runs it, the next turn carries the result
     "The user wants the current time. I have no clock of my own, but there is a clock tool in the list. I should call it and answer from the result.",
     "Let me check the clock.\n<tool>{\"name\": \"clock\", \"args\": {}}</tool>"),
    ("<result name=\"clock\">",
     "The clock tool returned the current date and time. I will state it plainly.",
     "Right now it is the time shown by the clock above."),
    ("What is in this folder?",   # a tool that asks first (shell is policy ask): the owner sees allow / deny before it runs
     "The user wants a listing of the current folder. The shell tool can do that; a plain directory listing is harmless.",
     "I will list it.\n<tool>{\"name\": \"shell\", \"args\": {\"cmd\": \"ls\"}}</tool>"),
    ("<result name=\"shell\">",
     "The listing came back. I will summarise what is there.",
     "The folder holds the client (chat.py, app.py and app/), the tool belt (tools/), the engine sources and the readme."),
]

class Replay:   # stands in for the engine: same protocol, canned turns, a chosen pace; nothing is loaded
    def __init__(self, cfg, rate):
        self.q = queue.Queue(); self.ready = False; self.ctx = int(cfg["ctx"]); self.pos = 0; self.rate = max(0.5, rate); self.n = 0; self.stop = False; self.p = None; self.scratch = []
        self.stats = collections.deque(maxlen=400)   # a synthetic stderr feed in the engine's own line formats, so the pace display can be seen without the engine
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
        self.stop = False; t0 = time.time(); tn = an = 0; self.pos += len(self.split(text)) + 2; emitted = [0]; passes = [0]
        def emit(kind, piece, tid=7):
            time.sleep(1.0 / self.rate); self.q.put(f"{kind} {tid} {esc(piece)}"); self.pos += 1; emitted[0] += 1
            if emitted[0] % 6 == 0:                                      # every few tokens, the lines the engine writes in chat mode, in its own shapes: a pass line and the running rate
                passes[0] += 1; drafted = (passes[0] % 3 == 0) * 5; secs = time.time() - t0
                if passes[0] % 7 == 0: self.stats.append(f", the  [geometry] 12 tokens from stored walk {int(t0)}@{self.pos}, no pass (0.01s)")
                else: self.stats.append(f"pass {passes[0]:3d}: {drafted} drafted, {drafted} accepted + 1 from the head | {1.20 if drafted else 1.33:.2f}s -> {(drafted + 1) / 1.2:.2f} tok/s this pass (experts read 0.9 upload 0.02 | compute 0.2 | disk 2.0 GB)")
                self.stats.append(f"  [live] {emitted[0]} tokens in {secs:.1f}s = {emitted[0] / max(secs, 0.01):.2f} tok/s")
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
        self.q.put(f"END {reason} {tn} {an} {time.time() - t0:.2f} {self.pos} {self.ctx}")   # like the engine in chat mode, no summary lines per turn: the client adds the turn up
    def get(self, timeout=0.25):
        try: return self.q.get(timeout=timeout)
        except queue.Empty: return ""
    wait_for = Engine.wait_for; cleanup = Engine.cleanup

# ---------------------------------------------------------------- the commands: the palette and /help come from this table
COMMANDS = [   # (name, arguments, what it does)
    ("/new", "", "next conversation; this one is kept"), ("/chats", "", "the last twelve conversations"), ("/open", "N", "continue a conversation from /chats or /find"),
    ("/find", "WORDS", "search every conversation"), ("/restart", "", "start the engine again after it stopped; the conversation continues"),
    ("/again", "", "ask the last message again; the old answer is kept as a branch"), ("/edit", "[N]", "edit the Nth-last message and continue from it; the old tail is kept as a branch"),
    ("/fast", "", "preset: small context, short think and answer, tools off"), ("/deep", "", "preset: full context, unlimited, tools on"),
    ("/think", "N", "think tokens per turn, 0 = unlimited"), ("/answer", "N", "answer tokens per turn, 0 = unlimited"), ("/bars", "N", "lines of thought under the hood"),
    ("/stats", "", "the speed line after each answer, on or off"), ("/ctx", "[N]", "context in use, or set it for the next start"),
    ("/sandbox", "[on|off]", "memory for this conversation only, nothing loaded or kept"), ("/memory", "", "which memory mode is on"),
    ("/tools", "[on|off|auto]", "the tool belt and each tool's policy"), ("/tool", "NAME allow|ask|off", "one tool's policy"), ("/hops", "N", "tool calls allowed per turn"),
    ("/rules", "[N|clear]", "remembered tool answers"), ("/cap", "N", "characters of a tool result the model sees"),
    ("/copy", "", "the last answer to the clipboard"), ("/queue", "[clear]", "messages waiting to send"), ("/quit", "", "leave; memory is saved"), ("/help", "", "this list"),
]

# ---------------------------------------------------------------- the full-screen chat
class TUI:
    def __init__(self, cfg, eng, keys, screen):
        self.cfg, self.eng, self.keys, self.screen = cfg, eng, keys, screen
        self.log = []            # {"kind": user|answer|dim|tool, "text": str, ...}
        self.inp = ""; self.caret = 0; self.scroll = 0; self.scrolled_at = 0; self.lifted = False; self.tscroll = 0; self.tmax = 0
        self.turn = None; self.history = []; self.dead = False; self.ctrlc = 0; self.flash = None; self.t_start = time.time()
        self.conv = None; self.carry_next = False; self.branches = []   # the conversation on disk; carry_next: a reopened or edited one continues from its last exchange; branches: tails cut by /again and /edit
        self.hist = []; self.hist_i = None; self.queue = []; self.pal_i = 0; self.pal_for = None   # sent messages for Up to recall; messages typed during a turn; the palette's selection
        self.lift_anim = None; self.hood_n = int(cfg["bars"]); self.draft_dirty = False; self.draft_at = 0
        self.pick = None; self.done_at = 0   # pick: the conversation ids /chats or /find listed, for /open N; done_at: when the last answer landed, for the title and the bell
        self.pace = None; self.pace_tally = None   # from the engine's stderr: the running rate and the kind of the last pass, for the hood; the turn added up, for one dim line after
        os.makedirs(os.path.join(HERE, "results", "chat"), exist_ok=True); os.makedirs(CONVS, exist_ok=True)
        self.logfile = os.path.join(HERE, "results", "chat", datetime.date.today().isoformat() + ".md")

    # ---- conversations: every one is a json file; /new keeps the old one, /chats lists them, /open N brings one back
    def save_conv(self):
        if not self.log: return
        if not self.conv:
            first = next((e["text"] for e in self.log if e["kind"] == "user"), "")
            self.conv = {"id": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "title": " ".join(first.split())[:60] or "untitled", "created": time.time()}
        self.conv["updated"] = time.time(); self.conv["turns"] = sum(1 for e in self.log if e["kind"] == "user")
        try:
            with open(os.path.join(CONVS, self.conv["id"] + ".json"), "w", encoding="utf-8") as f: json.dump({**self.conv, "log": self.log, "history": self.history, "branches": self.branches}, f, ensure_ascii=False)
        except Exception: pass
    def new_conv(self):
        self.save_conv(); self.conv = None; self.carry_next = False; self.branches = []
        if self.eng.ready: self.eng.send("R"); self.eng.pos = 0   # pos drops now so the next turn opens with the tool list even if RESET is still in flight
        self.log = []; self.history = []; self.scroll = 0
    def open_conv(self, cid):
        try:
            with open(os.path.join(CONVS, cid + ".json"), encoding="utf-8") as f: c = json.load(f)
        except Exception: return False
        self.save_conv(); self.conv = {k: c[k] for k in ("id", "title", "created", "updated", "turns") if k in c}
        self.log = c.get("log", []); self.history = c.get("history", [])[-1:]; self.branches = c.get("branches", []); self.scroll = 0; self.carry_next = True
        self.hist = [e["text"] for e in self.log if e["kind"] == "user"]; self.hist_i = None
        if self.eng.ready: self.eng.send("R"); self.eng.pos = 0
        return True
    @staticmethod
    def list_convs(n=50):
        found = []
        for f in sorted(os.listdir(CONVS), reverse=True)[:n]:
            try:
                with open(os.path.join(CONVS, f), encoding="utf-8") as fh: c = json.load(fh)
                found.append({**{k: c.get(k) for k in ("id", "title", "created", "updated", "turns")}, "branches": len(c.get("branches") or [])})
            except Exception: pass
        return found
    def branch_at(self, i):   # cut the log from entry i (a user message) on; the tail is kept as a branch; the next turn goes into a fresh context carrying what came before
        self.branches.append({"at": time.time(), "log": self.log[i:]}); self.log = self.log[:i]
        self.history = self.history[:sum(1 for e in self.log if e["kind"] == "user")]; self.carry_next = True; self.scroll = 0
        if self.eng.ready: self.eng.send("R"); self.eng.pos = 0

    # ---- the box: a draft that survives a restart, history, word jumps, kills
    def load_draft(self):
        try:
            with open(DRAFT, encoding="utf-8") as f: self.inp = f.read()
            self.caret = len(self.inp)
        except OSError: pass
    def save_draft(self, force=False):
        if not self.draft_dirty or (not force and time.time() - self.draft_at < 0.8): return
        self.draft_dirty = False; self.draft_at = time.time()
        try:
            if self.inp:
                with open(DRAFT, "w", encoding="utf-8") as f: f.write(self.inp)
            elif os.path.exists(DRAFT): os.remove(DRAFT)
        except OSError: pass
    def insert(self, s):
        self.inp = self.inp[:self.caret] + s + self.inp[self.caret:]; self.caret += len(s); self.hist_i = None; self.draft_dirty = True
    def set_box(self, s, caret=None):
        self.inp = s; self.caret = len(s) if caret is None else caret; self.draft_dirty = True
    def word_left(self):
        i = self.caret
        while i > 0 and not self.inp[i - 1].isalnum(): i -= 1
        while i > 0 and self.inp[i - 1].isalnum(): i -= 1
        return i
    def word_right(self):
        i, n = self.caret, len(self.inp)
        while i < n and not self.inp[i].isalnum(): i += 1
        while i < n and self.inp[i].isalnum(): i += 1
        return i
    def hist_move(self, d):   # Up on an empty box recalls what you sent, older each press; Down comes back
        if not self.hist: return
        if self.hist_i is None:
            if d > 0: return
            self.hist_i = len(self.hist) - 1
        else:
            i = self.hist_i + d
            if i >= len(self.hist): self.hist_i = None; self.set_box(""); return
            self.hist_i = max(0, i)
        self.set_box(self.hist[self.hist_i]); self.hist_i = self.hist_i   # set_box keeps the index
    def caret_row_move(self, d, inner):   # Up / Down inside a draft that spans rows
        segs = self.box_segs(inner); r = next((k for k, (s0, s) in enumerate(segs) if s0 <= self.caret <= s0 + len(s)), 0)
        if not (0 <= r + d < len(segs)): return False
        col = dw(self.inp[segs[r][0]:self.caret]); s0, s = segs[r + d]; self.caret = s0 + len(clip(s, col)); return True
    def attach(self, text):   # "@notes.md" or "@\"a path with spaces\"" in a message: the file's text goes to the model with it, cut at attach_cap; the log says what went
        user = text; notes = []
        for m in re.finditer(r'(?<!\S)@("([^"]+)"|(\S+))', text):
            raw = m.group(2) or m.group(3); path = os.path.expanduser(raw) if os.path.isabs(os.path.expanduser(raw)) else os.path.join(HERE, raw)
            if os.path.isdir(path): notes.append(f"@{raw} is a folder {G['dot']} name a file"); continue
            if not os.path.isfile(path): notes.append(f"no file @{raw}"); continue
            try:
                with open(path, encoding="utf-8", errors="replace") as f: body = f.read()
            except OSError as e: notes.append(f"@{raw}: {e}"); continue
            cap = int(self.cfg.get("attach_cap") or 6000); cut = len(body) > cap; body = body[:cap]
            user += f'\n\n<file name="{os.path.basename(path)}"{" cut=\"true\"" if cut else ""}>\n{body}\n</file>'
            notes.append(f"attached {os.path.basename(path)} {G['dot']} {len(body):,} chars" + (f" of {len(body) + 1:,}+ {G['dot']} cut at {cap:,}" if cut else ""))
        return user, notes
    def complete_path(self):   # Tab on an @word: the first file whose name starts with it (or the common prefix of several)
        m = re.search(r'(?<!\S)@("?)([^"\s]*)$', self.inp[:self.caret])
        if not m: return False
        raw = m.group(2); d, base = os.path.split(raw); folder = os.path.join(HERE, d) if not os.path.isabs(os.path.expanduser(d or ".")) else os.path.expanduser(d)
        try: names = sorted(n for n in os.listdir(folder or ".") if n.startswith(base) and not n.startswith("."))
        except OSError: return False
        if not names: return False
        pick = os.path.commonprefix(names) if len(names) > 1 else names[0]
        if len(names) == 1 and os.path.isdir(os.path.join(folder, names[0])): pick += "/"
        new = os.path.join(d, pick) if d else pick; quoted = " " in new
        start = m.start(); self.inp = self.inp[:start] + "@" + (f'"{new}"' if quoted else new) + self.inp[self.caret:]; self.caret = start + 1 + len(new) + (2 if quoted else 0); self.draft_dirty = True
        return True
    def done(self, secs):   # an answer landed: the title says so for a moment, and a bell if it took long enough that you may have walked away
        self.done_at = time.time()
        if secs >= float(self.cfg.get("bell_after", 15)) and self.cfg.get("bell", True): self.screen.bell()
    # ---- pace: what the engine says on stderr, shown truthfully. [live] is the running rate; "pass k: D drafted, A accepted" and
    #      "[geometry] N tokens from stored walk" say what kind of pass the last one was. In chat mode the engine writes no per-turn
    #      summary (its "geometry runs:" / "family drafts:" lines come once, at exit — fixtures/chat_engine_3turns_20261007.log), so the
    #      client adds the turn up itself from those lines and leaves one dim line under the turn. A [pass] breakdown can share a line with
    #      streamed text, so [geometry] is searched for, not anchored; the three shapes below are the real engine's, verbatim.
    PACE_LIVE = re.compile(r"^\s*\[live\] (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s")
    PACE_PASS = re.compile(r"^pass\s+(\d+): (\d+) drafted, (\d+) accepted \+ 1 from the head \| ([\d.]+)s")
    PACE_GEO = re.compile(r"\[geometry\] (\d+) tokens from stored walk")
    def pace_update(self):
        stats = getattr(self.eng, "stats", None)
        t = self.turn; tally = self.pace_tally
        if t and (not tally or tally["t0"] != t["t0"]) and (tally is None or tally["passes"] == 0): tally = self.pace_tally = {"t0": t["t0"], "passes": 0, "drafts": 0, "stored": 0, "cold": 0, "rate": 0.0}
        while stats:
            line = stats.popleft(); now = time.time()
            if m := self.PACE_LIVE.match(line):
                self.pace = {**(self.pace or {}), "rate": float(m.group(3)), "at": now}
                if tally: tally["rate"] = float(m.group(3))
            elif m := self.PACE_PASS.match(line):
                drafted, accepted = int(m.group(2)), int(m.group(3))
                self.pace = {**(self.pace or {}), "kind": "drafted" if accepted else ("cold" if not drafted else ""), "at": now}
                if tally: tally["passes"] += 1; tally["drafts"] += accepted; tally["cold"] += not drafted
            elif m := self.PACE_GEO.search(line):
                self.pace = {**(self.pace or {}), "kind": "stored", "at": now}
                if tally: tally["stored"] += int(m.group(1)); tally["passes"] += 1
        if tally and tally["passes"] and self.log and (not t or t["t0"] != tally["t0"]):   # the turn has ended: one dim line under it, even if a queued turn has already begun
            parts = [f"{tally['rate']:.1f} tok/s"] * bool(tally["rate"]) + [f"{tally['passes']} passes"]
            parts += [f"{tally['drafts']} tokens from drafts"] * bool(tally["drafts"]) + [f"{tally['stored']} from stored runs"] * bool(tally["stored"]) + [f"{tally['cold']} cold"] * bool(tally["cold"])
            note = {"kind": "dim", "text": f" {G['dot']} ".join(parts)}; self.pace_tally = None; self.pace = None
            at = len(self.log)
            if t:                                                        # a queued message already started the next turn: the line goes above it
                users = [i for i, e in enumerate(self.log) if e["kind"] == "user"]; at = users[-1] if users else at
            self.log.insert(at, note)
            if t: self.pace_tally = {"t0": t["t0"], "passes": 0, "drafts": 0, "stored": 0, "cold": 0, "rate": 0.0}
    def pace_text(self):   # " · 1.2 tok/s · drafted" for the hood's status line and the title, while the pace is fresh
        p = self.pace
        if not p or time.time() - p.get("at", 0) > 8: return ""
        s = (f" {G['dot']} {p['rate']:.1f} tok/s" if p.get("rate") else "") + (f" {G['dot']} {p['kind']}" if p.get("kind") else "")
        return s
    def palette(self):   # the commands matching what is typed, while the first word is still being typed
        s = self.inp
        if not s.startswith("/") or any(c in s for c in " \n"): return []
        return [c for c in COMMANDS if c[0].startswith(s)]
    def scroll_by(self, d):
        self.scroll = max(0, self.scroll + d); self.scrolled_at = time.time()

    # ---- keys
    def handle(self, k):
        t = self.turn
        if t and t.get("stage") == "ask" and k in ("y", "Y", "n", "N", "a", "A", "ENTER"): self.answer_tool(k in ("y", "Y", "ENTER"), always=k in ("a", "A")); return
        pal = self.palette()
        if pal and self.pal_for != self.inp: self.pal_i = 0; self.pal_for = self.inp
        if k == "CTRL_C":
            if t and t.get("stage") in ("tool", "ask"): self.abort_tool()
            elif t and not t["stopped"]: self.eng.send("X"); t["stopped"] = True
            elif not t:                                               # twice within two seconds quits; once is easy to hit by habit
                if time.time() - self.ctrlc < 2: raise Quit()
                self.ctrlc = time.time(); self.flash = ("Ctrl-C again to quit", time.time() + 2)
        elif k == "ENTER":
            if pal and self.inp.strip() != pal[self.pal_i][0]: self.set_box(pal[self.pal_i][0])   # a prefix takes the selected command
            text = self.inp.strip()
            if text.startswith("/") or not t: self.submit()
            elif text: self.queue.append(text); self.set_box(""); self.hist_i = None   # typed while the model answers: it goes the moment the turn ends
        elif k == "NL": self.insert("\n")
        elif k == "BS":
            if self.caret: self.inp = self.inp[:self.caret - 1] + self.inp[self.caret:]; self.caret -= 1; self.hist_i = None; self.draft_dirty = True
        elif k == "DEL": self.inp = self.inp[:self.caret] + self.inp[self.caret + 1:]; self.draft_dirty = True
        elif k == "LEFT": self.caret = max(0, self.caret - 1)
        elif k == "RIGHT": self.caret = min(len(self.inp), self.caret + 1)
        elif k == "HOME": self.caret = 0
        elif k == "END": self.caret = len(self.inp)
        elif k == "CTRL_LEFT": self.caret = self.word_left()
        elif k == "CTRL_RIGHT": self.caret = self.word_right()
        elif k == "CTRL_K":                                           # kill to the end of the line
            end = self.inp.find("\n", self.caret); end = len(self.inp) if end < 0 else end; self.inp = self.inp[:self.caret] + self.inp[end:]; self.draft_dirty = True
        elif k == "CTRL_U":                                           # kill to the start of the line
            start = self.inp.rfind("\n", 0, self.caret) + 1; self.inp = self.inp[:start] + self.inp[self.caret:]; self.caret = start; self.draft_dirty = True
        elif k == "CTRL_W":                                           # delete the word before the caret
            i = self.word_left(); self.inp = self.inp[:i] + self.inp[self.caret:]; self.caret = i; self.draft_dirty = True
        elif k in ("WHEEL_UP", "WHEEL_DOWN"): self.scroll_by(3 if k == "WHEEL_UP" else -3)
        elif k in ("UP", "PGUP", "DOWN", "PGDN"):
            d = (1 if k in ("UP", "DOWN") else 10) * (1 if k in ("UP", "PGUP") else -1)
            if pal and k in ("UP", "DOWN"): self.pal_i = (self.pal_i - d) % len(pal)   # the palette: arrows pick
            elif self.lifted and self.hood_text(): self.tscroll = min(self.tmax, max(0, self.tscroll + d))   # the open hood: scroll the thought
            elif k in ("UP", "DOWN") and (self.hist_i is not None or (not self.inp and self.hist)): self.hist_move(-d)   # an empty box: recall what you sent
            elif k in ("UP", "DOWN") and ("\n" in self.inp or len(self.box_segs(self.screen.size()[0] - 7)) > 1):   # a draft over several rows: move within it
                if not self.caret_row_move(-d, self.screen.size()[0] - 7): self.scroll_by(d)
            else: self.scroll_by(d)
        elif k == "TAB":
            if pal: name, args, _ = pal[self.pal_i]; self.set_box(name + (" " if args else "")); return   # the palette: Tab completes
            if self.complete_path(): return                                                               # an @file: Tab completes the path
            self.toggle_hood()
        elif k == "ESC":                                              # stops the turn when one is running, else clears the box, else the queue
            if t and t.get("stage") in ("tool", "ask"): self.abort_tool()
            elif t and not t["stopped"]: self.eng.send("X"); t["stopped"] = True
            elif self.inp: self.set_box(""); self.hist_i = None
            elif self.queue: self.queue = []; self.flash = ("queue cleared", time.time() + 2)
        elif k == "QUIT": raise Quit()
        else: self.insert(k)   # typed or pasted characters
    def toggle_hood(self):
        want = not self.lifted and bool(self.hood_text(lifted=True))
        if want == self.lifted: return
        self.lifted = want; self.tscroll = 0; self.lift_anim = [time.time(), self.hood_n, None]   # the target is known at draw time (it depends on the screen)
    def note(self, text, **extra): self.log.append({"kind": "dim", "text": text, **extra})
    def hood_text(self, lifted=None):   # what the hood shows: (head, text, live) for the thought in progress, a tool running or asking, or the last thought / result when opened over it
        t = self.turn
        if t and t.get("stage") == "tool": return f"{tools.describe(*t['tool'][:2])} {G['dot']} {fmt_t(time.time() - t['tool'][2])}", "", True
        if t and t.get("stage") == "ask": return f"run {tools.describe(*t['pending'])} ?  y / n / a = always", "", True
        if t and (t["in_think"] or t.get("hood_close")): return "thinking " + G["dot"] + " " + fmt_t(time.time() - t["t0"]) + self.pace_text() + (" " + G["dot"] + " stopping" if t["stopped"] else ""), t["think"][:t["think_shown"]], True
        return None                                                    # idle: nothing. A finished thought stays off; "thought for Ns" is all that remains (the TV rule)
    def abort_tool(self):
        t = self.turn; self.log.append({"kind": "tool", "text": tools.describe(*(t.get("pending") or t["tool"][:2])) + f" {G['dot']} stopped", "result": ""})
        self.note("stopped" + (f" {G['dot']} {t['hop']} tool call{'s' if t['hop'] != 1 else ''}" if t["hop"] else "")); self.history.append((t["msg"], t["parts"][-1][1] if t["parts"] else "")); self.turn = None; self.save_conv()
    def submit(self):
        text = self.inp.strip(); self.set_box(""); self.save_draft(force=True); self.scroll = 0; self.lifted = False; self.lift_anim = None; self.tscroll = 0; self.hist_i = None
        if not text: return
        if text.startswith("/"): self.command(text); return
        if self.dead: self.note("engine stopped " + G["dot"] + " see results/server_core/chat_engine.log"); return
        if not self.hist or self.hist[-1] != text: self.hist.append(text)
        self.log.append({"kind": "user", "text": text})
        user, notes = self.attach(text)                                  # @file words bring the file's text along with the message
        for n in notes: self.note(n)
        self.turn = {"msg": text, "user": user, "think": "", "ans": "", "think_shown": 0, "ans_shown": 0, "budget": 0.0, "arrivals": collections.deque(), "t_frame": time.time(),
                     "t0": time.time(), "in_think": True, "stopped": False, "stage": "ready?", "attempt": 0, "hop": 0, "tok": 0, "secs": 0.0, "parts": []}   # hop: tool round trips so far; parts: (thought, answer) per hop
        if self.eng.ready: self.send_turn()
    def send_turn(self):
        t = self.turn; think, answer = int(self.cfg["think"]), int(self.cfg["answer"]); pre = tools.preamble(self.cfg)
        if answer == 0: answer = 1 << 20                                    # unlimited: the engine build in use clamps 0 up to 1, so send "huge" instead
        est = len(t["user"]) // 3 + 8; need = est + min(think or 160, 160) + min(answer, 120)
        if t["attempt"] == 0 and not self.carry_next and self.eng.ctx - self.eng.pos >= need + (len(pre) // 3 if self.eng.pos == 0 else 0):
            text = (pre if self.eng.pos == 0 else "") + f"{U}{t['user']}{A}<think>\n"   # the tool list opens every fresh context
        elif t["attempt"] <= 1:                                      # will not fit after what is there (or a reopened conversation): start over, carrying the last exchange when it fits
            if t["stage"] != "reset": self.eng.send("R"); t["stage"] = "reset"; return
            reopened, self.carry_next = self.carry_next, False
            if t["hop"]:                                             # mid tool loop: fresh context with the question and the result together
                text = pre + f"{U}{t['msg']}\n\n{t['user']}{A}<think>\n"; carry = []
            else:
                carry = [(u, a) for u, a in self.history[-1:] if a and t["attempt"] == 0 and len(u + a) // 3 + 8 <= self.eng.ctx - need - len(pre) // 3]
                text = pre + "".join(f"{U}{u}{A}{a}{EOS}" for u, a in carry) + f"{U}{t['user']}{A}<think>\n"
            self.history = []
            if not reopened: self.log.insert(len(self.log) - (1 if self.log and self.log[-1]["kind"] == "user" else 0), {"kind": "dim", "text": f"context full {G['dot']} fresh start" + (f" {G['dot']} carrying the last exchange" if carry else "")})
        else:
            self.note("message is too long for the context"); self.turn = None; return
        self.eng.send(f"U {think} {answer} {esc(text)}"); t["stage"] = "sent"

    # ---- engine messages
    def pump(self):
        self.poll_tool(); self.smooth()
        while True:
            m = self.eng.get(0)
            if m is None: self.engine_died(); return
            if not m: break
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
                t["arrivals"].append((time.time(), len(piece)))
                if t["in_think"]:
                    if tid == THINK_CLOSE: self.close_hood()
                    elif tid != ENDTOK: t["think"] += piece
                elif tid != ENDTOK:
                    t["ans"] += piece
                    if "</tool>" in t["ans"] and not t["stopped"] and tools.enabled(self.cfg):   # the call is closed: stop the engine now instead of letting it run on
                        self.eng.send("X"); t["stopped"] = True; t["tool_stop"] = True
            elif head == "END" and self.turn: self.finish(rest.split())
        if not self.turn and self.queue and self.eng.ready and not self.dead: self.set_box(self.queue.pop(0)); self.submit()   # the next queued message goes
    def smooth(self):   # text is shown at a steady pace: the engine hands over bursts, the screen releases characters at the measured rate (faster when far behind)
        self.pace_update(); t = self.turn; now = time.time()
        if not t: return
        dt = now - t["t_frame"]; t["t_frame"] = now; arr = t["arrivals"]
        while arr and arr[0][0] < now - 4: arr.popleft()
        rate = max(12.0, sum(n for _, n in arr) / 4.0)
        for key in ("think", "ans"):
            full, shown = t[key], t[key + "_shown"]
            if shown < len(full):
                lag = len(full) - shown; t["budget"] += rate * (3.0 if lag > rate * 1.5 else 1.0) * dt
                n = min(lag, int(t["budget"])); t["budget"] -= n; t[key + "_shown"] = shown + n
            else: t["budget"] = min(t["budget"], 2.0)
        if t.get("hood_close") and now - t["hood_close"] >= HOOD_CLOSE: self.hood_closed()
        if t["ans"] and not t.get("hood_close") and not t["in_think"]:   # the answer entry appears once the hood has dropped, and grows at the shown pace
            if not t.get("ans_open"): self.log.append({"kind": "answer", "text": ""}); t["ans_open"] = True
            if self.log and self.log[-1]["kind"] == "answer": self.log[-1]["text"] = t["ans"][:t["ans_shown"]].lstrip("\n")
    def engine_died(self):   # the engine process ended: say so, show the tail of its log folded under the line, and offer /restart; the conversation stays
        self.dead = True; tail = ""
        try:
            with open(ENGINE_LOG, encoding="utf-8", errors="replace") as f: tail = "".join(f.readlines()[-30:]).strip()
        except OSError: pass
        if self.turn: self.history.append((self.turn["msg"], self.turn["parts"][-1][1] if self.turn["parts"] else "")); self.turn = None; self.save_conv()
        self.note("engine stopped " + G["dot"] + " /restart starts it again" + (" " + G["dot"] + " Tab shows its log" if tail else ""), result=tail or "(no log at " + os.path.relpath(ENGINE_LOG, HERE) + ")")
    def close_hood(self):   # the thought is complete: everything shown, then the hood drops over it on a curve
        t = self.turn; self.lifted = False; self.lift_anim = None; self.tscroll = 0
        t["think_shown"] = len(t["think"]); t["in_think"] = False; t["hood_close"] = time.time()
    def hood_closed(self):
        t = self.turn; t.pop("hood_close", None); self.note(f"thought for {fmt_t(time.time() - t['t0'])}", thought=t["think"].strip())
    def finish(self, end):
        t = self.turn; reason, think_n, answer_n, secs, pos, ctx = end[0], int(end[1]), int(end[2]), float(end[3]), int(end[4]), int(end[5])
        self.eng.pos, self.eng.ctx = pos, ctx
        if t["in_think"]: self.close_hood()
        if t.get("hood_close"): self.hood_closed()
        t["think_shown"] = len(t["think"]); t["ans_shown"] = len(t["ans"]); self.smooth()   # whatever is left lands now
        answer = t["ans"].strip(); thought = t["think"].strip(); notes = []
        t["tok"] += think_n + answer_n; t["secs"] += secs; t["parts"].append((thought, answer))
        tool_stop = t.pop("tool_stop", None); t["stopped"] = False     # tool_stop: we stopped the engine ourselves at </tool>, that is not the owner stopping
        call = tools.parse_call(answer) if tools.enabled(self.cfg) and (reason in ("eos", "cap") or tool_stop) else None
        if call and t["hop"] < int(self.cfg.get("hops", 4)):             # the answer ends in a tool call: run it and let the model carry on
            name, args, shown = call
            if self.log and self.log[-1]["kind"] == "answer":
                if shown: self.log[-1]["text"] = shown
                else: self.log.pop()
            if tools.decide(name, args, self.cfg) == "ask": t["stage"] = "ask"; t["pending"] = (name, args); return
            self.run_tool(name, args); return
        if reason == "stop" and not tool_stop: notes.append("stopped")
        elif reason == "full": notes.append("context full" if answer else ("ran out of room while thinking " + G["dot"] + " /think N forces an answer, /ctx higher gives it room" if not int(self.cfg["think"]) else "ran out of room while thinking " + G["dot"] + " /think lower or /ctx higher"))
        elif not answer: notes.append("no answer " + G["dot"] + " /answer higher")
        if call: notes.append(f"stopped after {t['hop']} tool calls {G['dot']} /hops higher")
        if self.cfg["stats"] and t["tok"] and t["secs"] > 0: notes.append(f"{t['tok'] / t['secs']:.1f} tok/s {G['dot']} {t['tok']} tokens {G['dot']} {fmt_t(t['secs'])}" + (f" {G['dot']} {t['hop']} tool call{'s' if t['hop'] != 1 else ''}" if t["hop"] else ""))
        if notes: self.note(f" {G['dot']} ".join(notes))
        self.history.append((t["msg"], answer)); self.turn = None; self.save_conv(); self.done(t["secs"])
        try:
            with open(self.logfile, "a", encoding="utf-8") as f:
                body = "\n\n".join(f"> {th.replace(chr(10), chr(10) + '> ')}\n\n**r1:** {an}" for th, an in t["parts"])
                f.write(f"\n### {datetime.datetime.now():%H:%M}\n\n**you:** {t['msg']}\n\n{body}\n")
        except Exception: pass

    # ---- tools: run one, show the dim line, hand the result back to the model as the next user turn
    def run_tool(self, name, args):
        t = self.turn; t.pop("pending", None); t["stage"] = "tool"; t["tool"] = (name, args, time.time()); t["tool_done"] = None
        threading.Thread(target=lambda: t.__setitem__("tool_done", tools.run(name, args, self.cfg)), daemon=True).start()
    def answer_tool(self, yes, always=False):   # the owner's y / n / a on a tool that asks first; a = yes, and remember it for this command
        t = self.turn
        if not t or t.get("stage") != "ask": return
        name, args = t["pending"]
        if always: self.flash = ("always: " + tools.remember(name, args, self.cfg), time.time() + 3); save_settings(self.cfg)
        if yes or always: self.run_tool(name, args); return
        t.pop("pending", None); self.log.append({"kind": "tool", "text": tools.describe(name, args) + f" {G['dot']} declined", "result": ""})
        self.continue_turn(tools.wrap_result(tools.Result(name, args, "the owner declined this call; answer without it", ok=False)))
    def poll_tool(self):
        t = self.turn
        if not t or t.get("stage") != "tool" or t.get("tool_done") is None: return
        r = t.pop("tool_done"); name, args, t0 = t.pop("tool")
        self.log.append({"kind": "tool", "text": f"{tools.describe(name, args)} {G['dot']} {fmt_t(r.secs)}" + (f" {G['dot']} {r.summary}" if r.summary else ""), "result": r.text})
        self.continue_turn(tools.wrap_result(r))
    def continue_turn(self, user):
        t = self.turn; t["hop"] += 1; t["user"] = user; t["think"] = ""; t["ans"] = ""; t["think_shown"] = 0; t["ans_shown"] = 0; t["budget"] = 0.0; t.pop("ans_open", None)
        t["t0"] = time.time(); t["in_think"] = True; t["attempt"] = 0; t["stage"] = "ready?"
        if self.eng.ready: self.send_turn()

    # ---- commands
    def command(self, line):
        parts = line.split(); c = parts[0].lower(); arg = parts[1] if len(parts) > 1 else None; dot = f" {G['dot']} "
        if c in ("/quit", "/q", "/exit"): raise Quit()
        if c == "/new":
            if self.turn: self.note("wait for the answer, or Ctrl-C"); return
            self.new_conv(); return
        if c == "/chats":
            convs = self.list_convs(12); self.pick = [x["id"] for x in convs]
            self.note("\n".join(f"{i + 1}. {datetime.datetime.fromtimestamp(x['updated'] or 0):%m-%d %H:%M}  {x['title']} {dot}{x['turns']} turn{'s' if x['turns'] != 1 else ''}" for i, x in enumerate(convs)) + "\n/open N brings one back" if convs else "no conversations yet"); return
        if c == "/find":
            words = line.split(" ", 1)[1].strip() if " " in line else ""
            if not words: self.note("/find WORDS " + G["dot"] + " searches every conversation"); return
            hits = self.find_convs(words); self.pick = [h["id"] for h in hits]
            self.note("\n".join(f"{i + 1}. {datetime.datetime.fromtimestamp(h['updated'] or 0):%m-%d}  {h['title']}{dot}{h['snippet']}" for i, h in enumerate(hits)) + "\n/open N brings one back" if hits else f"nothing has {words}"); return
        if c == "/open":
            if self.turn: self.note("wait for the answer, or Ctrl-C"); return
            ids = self.pick if self.pick is not None else [x["id"] for x in self.list_convs(12)]
            if not arg or not arg.isdigit() or not (0 < int(arg) <= len(ids)): self.note("/open N " + G["dot"] + " see /chats or /find"); return
            self.open_conv(ids[int(arg) - 1]); self.note("continuing" + dot + "the last exchange is carried into a fresh context"); return
        if c == "/restart":
            if self.turn: self.note("wait for the answer, or Ctrl-C"); return
            self.restart_engine(keep=True); self.note("engine starting" + dot + "the conversation continues, the last exchange carried"); return
        if c in ("/again", "/edit"):
            if self.turn: self.note("wait for the answer, or Ctrl-C"); return
            users = [i for i, e in enumerate(self.log) if e["kind"] == "user"]; n = int(arg) if c == "/edit" and arg and arg.isdigit() else 1
            if not users or n < 1 or n > len(users): self.note("nothing to " + ("ask again" if c == "/again" else "edit") + (f" {dot} /edit N, N up to {len(users)}" if users else "")); return
            i = users[-n]; msg = self.log[i]["text"]; self.branch_at(i)
            if c == "/again": self.flash = ("asking again" + dot + "the old answer is kept as a branch", time.time() + 3); self.set_box(msg); self.submit()
            else: self.set_box(msg); self.flash = ("editing" + dot + "the old tail is kept as a branch", time.time() + 3)
            return
        if c in ("/fast", "/deep"):
            p = dict((self.cfg.get("presets") or DEFAULTS["presets"]).get(c[1:], {}))
            if not p: self.note(f"no preset {c[1:]} in the settings"); return
            restart = p.get("ctx") is not None and int(p["ctx"]) != int(self.cfg["ctx"]); self.cfg.update(p); save_settings(self.cfg)
            self.note(c[1:] + dot + dot.join(f"{k} {'unlimited' if k in ('think', 'answer') and not v else v}" for k, v in p.items()) + (dot + "restarting the engine for the context" if restart else ""))
            if restart:
                if self.turn: self.note("the context applies at the next start (a turn is running)")
                else: self.restart_engine()
            return
        if c in ("/think", "/answer", "/bars"):
            key = c[1:]
            if arg is None or not arg.isdigit(): self.note(f"{key} {self.cfg[key]}" + (" (unlimited)" if key in ("think", "answer") and not self.cfg[key] else "")); return
            self.cfg[key] = max(0, int(arg)); save_settings(self.cfg); self.note(f"{key} {self.cfg[key]}" + (" (unlimited)" if key in ("think", "answer") and not self.cfg[key] else "")); return
        if c == "/stats":
            self.cfg["stats"] = not self.cfg["stats"]; save_settings(self.cfg); self.note("stats " + ("on" if self.cfg["stats"] else "off")); return
        if c in ("/sandbox", "/memory"):
            if c == "/memory" and arg is None: self.note("memory " + self.cfg.get("memory", "shared")); return
            if self.turn: self.note("wait for the answer, or Ctrl-C"); return
            want = {"on": "sandbox", "off": "shared", "sandbox": "sandbox", "shared": "shared"}.get(arg, None) if arg else ("shared" if self.cfg.get("memory") == "sandbox" else "sandbox")
            if want is None: self.note("/sandbox [on|off]"); return
            self.cfg["memory"] = want; save_settings(self.cfg); self.restart_engine()
            self.note("memory sandbox" + dot + "this conversation only, nothing kept" if want == "sandbox" else "memory shared" + dot + "grows across conversations"); return
        if c == "/ctx":
            if arg and arg.isdigit(): self.cfg["ctx"] = max(64, int(arg)); save_settings(self.cfg); self.note(f"ctx {self.cfg['ctx']} from the next start"); return
            self.note(f"{self.eng.pos} of {self.eng.ctx} used"); return
        if c == "/tools":
            self.cfg["tools"] = True if arg == "on" else False if arg == "off" else "auto" if arg == "auto" else self.cfg.get("tools", "auto"); save_settings(self.cfg)
            state = "on" if tools.enabled(self.cfg) else ("off (auto: ctx under 2048)" if self.cfg["tools"] == "auto" else "off")
            self.note(f"tools {state}{dot}" + "  ".join(f"{n} {tools.policy(n, self.cfg)}" for n in tools.REGISTRY) + f"\n/tool NAME allow|ask|off   /hops {self.cfg.get('hops', 4)}   /tools on|off|auto"); return
        if c == "/tool":
            mode = parts[2] if len(parts) > 2 else None
            if arg not in tools.REGISTRY or mode not in ("allow", "ask", "off"): self.note("/tool NAME allow|ask|off"); return
            self.cfg.setdefault("tool_policy", {})[arg] = mode; save_settings(self.cfg); self.note(f"{arg} {mode}"); return
        if c == "/hops":
            if arg and arg.isdigit(): self.cfg["hops"] = max(0, int(arg)); save_settings(self.cfg)
            self.note(f"hops {self.cfg.get('hops', 4)}"); return
        if c == "/rules":
            rules = self.cfg.get("tool_rules") or []
            if arg == "clear": self.cfg["tool_rules"] = []; save_settings(self.cfg); self.note("rules cleared"); return
            if arg and arg.isdigit() and 0 < int(arg) <= len(rules): rules.pop(int(arg) - 1); save_settings(self.cfg); self.note("rule dropped"); return
            self.note(("\n".join(f"{i + 1}. {r['tool']} {r['match']} {G['arrow']} {r['policy']}" for i, r in enumerate(rules)) + "\n/rules N drops one " + G["dot"] + " /rules clear") if rules
                      else "no rules" + dot + "answer a when a tool asks and it is remembered for that command"); return
        if c == "/cap":
            if arg and arg.isdigit(): self.cfg["tool_cap"] = max(200, int(arg)); save_settings(self.cfg)
            self.note(f"tool results cut at {self.cfg.get('tool_cap', 4000)} characters"); return
        if c == "/copy":
            last = next((e["text"] for e in reversed(self.log) if e["kind"] == "answer"), None)
            if not last: self.note("nothing to copy"); return
            self.note("copied" if clipboard(last) else "no clipboard tool (clip / xclip / wl-copy / pbcopy)"); return
        if c == "/queue":
            if arg == "clear": self.queue = []; self.note("queue cleared"); return
            self.note("\n".join(f"{i + 1}. {q}" for i, q in enumerate(self.queue)) if self.queue else "nothing queued" + dot + "Enter while the model answers queues a message"); return
        if c == "/help":
            self.note("\n".join(f"{(n + ' ' + a).strip():<28}{d}" for n, a, d in COMMANDS) + "\n" +
                      "Tab  open / close the hood, or complete a command   Up / Down  history, or a draft's rows, or scroll   Ctrl-←/→  by word   Ctrl-K / U  kill to end / start\n"
                      "Ctrl-J  new line in the box   Esc / Ctrl-C  stop the answer   y / n / a  when a tool asks   wheel  scroll"); return
        self.note("no such command " + G["dot"] + " /help")
    def restart_engine(self, keep=False):   # the memory mode and the context are fixed at engine start: stop this engine (it saves what it should), start a fresh one. keep: the conversation stays and continues from its last exchange
        self.save_conv(); self.quit(quiet=True); self.dead = False; self.t_start = time.time()
        if keep: self.history = self.history[-1:]; self.carry_next = True
        else: self.conv = None; self.carry_next = False; self.branches = []; self.log = []; self.history = []; self.scroll = 0
        self.eng = Replay(self.cfg, self.eng.rate) if isinstance(self.eng, Replay) else Engine(self.cfg)
    @staticmethod
    def find_convs(words, n=12):   # every conversation whose title or text has all the words; a snippet around the first
        want = [w.lower() for w in words.split()]; hits = []
        for f in sorted(os.listdir(CONVS), reverse=True):
            try:
                with open(os.path.join(CONVS, f), encoding="utf-8") as fh: c = json.load(fh)
            except Exception: continue
            text = c.get("title", "") + "\n" + "\n".join(e.get("text", "") for e in c.get("log", []) if e.get("kind") in ("user", "answer")); low = text.lower()
            if all(w in low for w in want):
                i = low.find(want[0]); a = max(0, i - 40); b = min(len(text), i + 60)
                hits.append({"id": c["id"], "title": c.get("title", ""), "updated": c.get("updated"), "turns": c.get("turns"), "snippet": " ".join(text[a:b].split())})
                if len(hits) >= n: break
        return hits

    # ---- drawing
    def log_lines(self, w):
        lines = []
        for e in self.log:
            if e["kind"] == "user": lines += [""] + [(DIM + G["prompt"] + " " + RESET + l if i == 0 else "  " + l) for i, l in enumerate(wrap(e["text"], w - 4))]
            elif e["kind"] == "answer": lines += ["  " + l for l in wrap(e["text"], w - 4)] if e["text"] else []
            else: lines += [DIM + "  " + l + RESET for l in wrap(e["text"], w - 4)]   # dim notes and → tool lines alike
        return lines
    def hood_rows(self, w, avail):   # the hood over the thought: a dim status line with the timer, then the last `bars` lines of thought; nothing drawn around it
        ht = self.hood_text()
        if not ht: self.hood_n = int(self.cfg["bars"]); return []
        head, text, live = ht; t = self.turn; now = time.time(); bars = int(self.cfg["bars"]); full = max(0, avail - 2)
        if t and t.get("hood_close"): n = round(bars * (1 - ease_out((now - t["hood_close"]) / HOOD_CLOSE))); open_ = False   # dropping over a finished thought
        elif self.lift_anim:                                             # opening or closing on Tab
            start, a, b = self.lift_anim
            if b is None: b = self.lift_anim[2] = full if self.lifted else bars
            p = ease_out((now - start) / HOOD_OPEN); n = round(a + (b - a) * p); open_ = self.lifted
            if p >= 1: self.lift_anim = None
        else: n = full if self.lifted else bars; open_ = self.lifted
        self.hood_n = n
        all_lines = flow(text, w - 5) if live else list(wrap(text, w - 5))
        if not open_: all_lines = [l for l in all_lines if l.strip()]              # under the hood every row is text; paragraph gaps only show when it is open
        self.tmax = max(0, len(all_lines) - n); self.tscroll = min(self.tscroll, self.tmax)
        end = len(all_lines) - (self.tscroll if open_ else 0)
        lines = all_lines[max(0, end - n):end] if n else []; lines += [""] * (n - len(lines))
        thumb = round((end - n) / self.tmax * (n - 1)) if open_ and self.tmax and n > 1 else None   # the scroll thumb, only when there is more than fits
        rows = [DIM + "  " + head + RESET]
        for i, l in enumerate(lines): rows.append(DIM + "  " + pad(l, w - 4) + RESET + (DIM + G["thumb"] + RESET if i == thumb else " "))
        return rows
    def box_segs(self, inner):   # the draft cut into rows of the box: (start offset, text) each; wraps at the last space that fits, else hard, by display width
        text = self.inp; segs = []; start = 0
        while True:
            nl = text.find("\n", start); end = nl if nl >= 0 else len(text)
            while dw(text[start:end]) > inner:
                fit = len(clip(text[start:end], inner)); cut = text.rfind(" ", start, start + fit + 1)
                cut = cut + 1 if cut > start else start + fit
                segs.append((start, text[start:cut])); start = cut
            segs.append((start, text[start:end]))
            if nl < 0: break
            start = nl + 1
        return segs
    def box_rows(self, w):   # the input box: one row per line, grows as you type; returns rows and the caret's (row, col)
        inner = w - 7; segs = self.box_segs(inner); pos = (0, 0)       # border, space, mark, space, text, space, border = w - 1 columns, the same as the corners
        for r, (s0, s) in enumerate(segs):                              # a caret on a wrap boundary sits at the start of the next row
            if s0 <= self.caret <= s0 + len(s): pos = (r, min(dw(self.inp[s0:self.caret]), inner))
        segs = [(s0, s[:-1] if s.endswith(" ") and dw(s) > inner else s) for s0, s in segs]   # the wrap space stays in the text, not in the box
        mark = (RESET if self.turn and (time.time() % 1.4) < 0.45 else DIM) + G["prompt"] + RESET   # the mark pulses while the model works, so a slow pass never looks like a hang
        top = DIM + G["tl"] + G["h"] * (w - 3) + G["tr"] + RESET; bot = DIM + G["bl"] + G["h"] * (w - 3) + G["br"] + RESET
        rows = [top] + [DIM + G["v"] + RESET + (" " + mark + " " if k == 0 else "   ") + pad(s, inner) + " " + DIM + G["v"] + RESET for k, (_, s) in enumerate(segs)] + [bot]
        return rows, pos
    def extra_rows(self, w):   # between the hood and the box: queued messages, the palette, a passing line, the loading line
        rows = [DIM + "  queued " + G["dot"] + " " + clip(q, w - 12) + RESET for q in self.queue]
        pal = self.palette()
        if pal:
            top = max(0, min(self.pal_i - 3, len(pal) - 7))
            for i, (n, a, d) in enumerate(pal[top:top + 7], top):
                row = clip("  " + (n + " " + a).strip().ljust(26) + d, w - 1)   # a narrow terminal cuts the description, never the name
                rows.append(row if i == self.pal_i else DIM + row + RESET)
        if self.flash and time.time() < self.flash[1]: rows.append(DIM + "  " + self.flash[0] + RESET)
        else: self.flash = None
        if not self.eng.ready and not self.dead: rows.append(DIM + "  loading the model " + G["dot"] + " " + fmt_t(time.time() - self.t_start) + RESET)
        return rows
    def render(self):
        w, h = self.screen.size(); box, (crow, ccol) = self.box_rows(w); extra = self.extra_rows(w); hood = self.hood_rows(w, h - len(box) - len(extra))
        log_h = max(1, h - len(box) - len(hood) - len(extra)); lines = self.log_lines(w)
        self.scroll = min(self.scroll, max(0, len(lines) - log_h))
        end = len(lines) - self.scroll; vis = lines[max(0, end - log_h):end]
        if self.scroll and self.turn and vis: vis = vis[:-1] + [DIM + f"  {G['down']} {self.scroll} more" + RESET]   # scrolled up while it streams: say how far
        area = [""] * (log_h - len(vis)) + vis
        if len(lines) > log_h and (self.scroll or time.time() - self.scrolled_at < 1.5):   # a hairline on the right edge says where you are, while you scroll
            size = max(1, round(log_h * log_h / len(lines))); top = round((len(lines) - self.scroll - log_h) / (len(lines) - log_h) * (log_h - size))
            for r in range(top, min(log_h, top + size)): area[r] = pad(area[r], w - 1) + DIM + G["bar"] + RESET
        rows = area + hood + extra + box
        t = self.turn; title = "loki" + (f" {G['dot']} " + ("thinking " + fmt_t(time.time() - t["t0"]) if t["in_think"] else "answering") + self.pace_text() if t else (f" {G['dot']} done" if time.time() - self.done_at < 6 else ""))
        self.screen.draw(rows, (ccol + 5, h - len(box) + 2 + crow), title)

    def run(self, demo=None):
        try:
            self.load_draft()
            if demo: self.set_box(demo); self.submit()
            while True:
                try:
                    for k in self.keys.poll(1.0 / FPS): self.handle(k)   # a frame clock: keys, the engine, the smoothing and one repaint, thirty times a second
                    self.pump(); self.render(); self.save_draft()
                except KeyboardInterrupt: self.handle("CTRL_C")
        except Quit: pass
        finally:
            self.save_draft(force=True); self.keys.close(); self.screen.close(); self.quit()
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
        self.eng.send(f"U {think} {answer} {esc(U + msg + A + '<think>' + chr(10))}"); in_think = True; ans = []
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
                elif tid != ENDTOK: ans.append(piece); out(piece)
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
    if "--ascii" in argv: term.use_glyphs(False)
    if keys is not None or frames:                                   # headless: the real glyphs unless --ascii, the frames file is what the tests read
        TUI(cfg, eng, ScriptedKeys(keys or ""), Screen(frames, tuple(int(v) for v in size.split("x")) if size else None)).run(demo)
    elif sys.stdin.isatty() and sys.stdout.isatty():
        if "--ascii" not in argv: term.use_glyphs(term.can_draw())
        mouse = "--no-mouse" not in argv and bool(cfg.get("mouse", True))
        TUI(cfg, eng, ConsoleKeys(mouse=mouse), Screen(mouse=mouse)).run(demo)
    else:
        LineChat(cfg, eng).run(demo)
