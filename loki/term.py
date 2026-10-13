# term.py -- the terminal under loki. Glyphs with a plain-ASCII fallback for terminals that cannot draw them, display
#   widths (CJK and emoji are two columns, combining marks none), width-aware wrapping, a screen that repaints only the
#   rows that changed, a key reader with word jumps and the mouse wheel, scripted keys for the headless tests, easing.
import os, sys, re, time, shutil, unicodedata, functools

DIM, RESET = "\x1b[2m", "\x1b[0m"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

def out(s=""):
    sys.stdout.write(s); sys.stdout.flush()

# ---------------------------------------------------------------- glyphs
UNICODE = {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│", "prompt": "›", "dot": "·", "thumb": "●", "arrow": "→", "down": "↓", "bar": "▏"}
ASCII = {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|", "prompt": ">", "dot": ".", "thumb": "*", "arrow": "->", "down": "v", "bar": "|"}
G = dict(UNICODE)

def can_draw():   # whether the terminal is likely to have the rounded corners and marks; conhost with a raster font and non-UTF-8 shells do not
    if os.environ.get("LOKI_ASCII"): return False
    if os.environ.get("TERM") == "dumb": return False
    if os.name == "nt": return bool(os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM") or os.environ.get("ConEmuANSI"))
    return "utf" in (getattr(sys.stdout, "encoding", "") or "").lower() or "utf" in (os.environ.get("LANG", "") + os.environ.get("LC_ALL", "")).lower()

def use_glyphs(unicode_ok):
    G.clear(); G.update(UNICODE if unicode_ok else ASCII)

# ---------------------------------------------------------------- widths
ZERO = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF}
@functools.lru_cache(maxsize=4096)
def cw(ch):   # columns one character takes
    o = ord(ch)
    if o < 32 or 0x7F <= o < 0xA0: return 0
    if o < 0x300: return 1
    if o in ZERO or 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF or unicodedata.combining(ch): return 0
    if unicodedata.east_asian_width(ch) in ("W", "F") or 0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF and unicodedata.east_asian_width(ch) == "W": return 2
    return 1

def dw(s):   # display width of a string (ANSI escapes count for nothing)
    return sum(cw(c) for c in (ANSI.sub("", s) if "\x1b" in s else s))

def clip(s, w):   # the longest prefix that fits in w columns
    n = 0
    for i, c in enumerate(s):
        n += cw(c)
        if n > w: return s[:i]
    return s

@functools.lru_cache(maxsize=2048)
def wrap(text, w):   # word wrap by display width; paragraphs on "\n"; a word wider than a row is cut
    lines = []
    for para in text.split("\n"):
        cur = ""; cur_w = 0
        for word in para.split(" "):
            ww = dw(word)
            if ww > w:
                if cur: lines.append(cur); cur = ""; cur_w = 0
                while ww > w: head = clip(word, w) or word[:1]; lines.append(head); word = word[len(head):]; ww = dw(word)
            need = ww + (1 if cur else 0)
            if cur_w + need <= w: cur = (cur + " " + word) if cur else word; cur_w += need
            else: lines.append(cur); cur = word; cur_w = ww
        lines.append(cur.rstrip())
    return lines

def flow(text, w):   # wrap for text still arriving: the unfinished last word stays on its row (clipped) until a space lands, so rows fill left to right and never reflow
    cut = max(text.rfind(" "), text.rfind("\n")) + 1; lines = list(wrap(text[:cut], w)); tail = text[cut:]
    if tail: lines[-1] = clip(lines[-1] + (" " if lines[-1] and not text[:cut].endswith("\n") else "") + tail, w) if lines else clip(tail, w)
    return lines

def pad(s, w):   # pad with spaces to w columns
    return s + " " * max(0, w - dw(s))

# ---------------------------------------------------------------- easing and a frame clock
def ease_out(t): t = max(0.0, min(1.0, t)); return 1 - (1 - t) ** 3
def ease_in_out(t): t = max(0.0, min(1.0, t)); return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
FPS = 30

# ---------------------------------------------------------------- the screen
class Screen:
    def __init__(self, frames=None, size=None, mouse=True):
        self.frames = open(frames, "w", encoding="utf-8") if frames else None; self.size_fixed = size; self.n = 0; self.live = not frames
        self.prev = None; self.prev_cursor = None; self.prev_title = None; self.prev_size = None; self.mouse = mouse and self.live
        if self.live: out("\x1b[?1049h\x1b[?25l\x1b[H\x1b[2J" + ("\x1b[?1006h\x1b[?1000h" if self.mouse else ""))   # alternate screen; SGR mouse reporting for the wheel
    def size(self):
        if self.size_fixed: return self.size_fixed
        s = shutil.get_terminal_size((100, 30)); return max(40, s.columns), max(12, s.lines)
    def draw(self, rows, cursor, title="loki"):
        w, h = self.size(); rows = [r for r in rows[:h]] + [""] * (h - len(rows))
        resized = self.prev_size != (w, h)
        if not resized and rows == self.prev and cursor == self.prev_cursor and title == self.prev_title: return
        if self.live:
            s = "\x1b[?25l" + (f"\x1b]0;{title}\x07" if title != self.prev_title else "") + ("\x1b[2J" if resized else "")
            for i, r in enumerate(rows):                                  # only the rows that changed are sent
                if resized or self.prev is None or i >= len(self.prev) or self.prev[i] != r: s += f"\x1b[{i + 1};1H{r}\x1b[K"
            s += f"\x1b[{cursor[1]};{cursor[0]}H\x1b[?25h"; out(s)
        else:
            self.n += 1; self.frames.write(f"---- frame {self.n} ({w}x{h}) cursor {cursor}\n" + "\n".join(ANSI.sub("", r) for r in rows) + "\n"); self.frames.flush()
        self.prev, self.prev_cursor, self.prev_title, self.prev_size = rows, cursor, title, (w, h)
    def bell(self):   # the terminal bell; the frames file notes it so a test can see it rang
        if self.live: out("\a")
        else: self.frames.write("---- bell\n"); self.frames.flush()
    def close(self):
        if self.live: out(("\x1b[?1000l\x1b[?1006l" if self.mouse else "") + "\x1b[?25h\x1b[?1049l")
        else: self.frames.close()

# ---------------------------------------------------------------- keys
class Quit(Exception): pass

CTRL = {"\r": "ENTER", "\n": "NL", "\t": "TAB", "\x08": "BS", "\x7f": "BS", "\x1b": "ESC", "\x03": "CTRL_C", "\x04": "CTRL_C",
        "\x0b": "CTRL_K", "\x15": "CTRL_U", "\x17": "CTRL_W", "\x01": "HOME", "\x05": "END"}
ESC = {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT", "[H": "HOME", "[F": "END", "[1~": "HOME", "[4~": "END", "[3~": "DEL", "[5~": "PGUP", "[6~": "PGDN", "OH": "HOME", "OF": "END",
       "[1;5C": "CTRL_RIGHT", "[1;5D": "CTRL_LEFT", "[1;3C": "CTRL_RIGHT", "[1;3D": "CTRL_LEFT", "[1;5A": "UP", "[1;5B": "DOWN"}
SEQ = re.compile(r"\x1b(\[<\d+;\d+;\d+[mM]|\[[0-9;]*[A-Za-z~]|O[A-Z])")

def parse_escapes(data):   # the escape sequences in a chunk of terminal input: keys, and SGR mouse events for the wheel
    keys = []; i = 0
    while i < len(data):
        c = data[i]
        if c == "\x1b":
            m = SEQ.match(data, i)
            if m:
                s = m.group(1); i = m.end()
                if s.startswith("[<"):
                    b = int(s[2:].split(";")[0])
                    if b in (64, 65) and s.endswith("M"): keys.append("WHEEL_UP" if b == 64 else "WHEEL_DOWN")
                else: keys.append(ESC.get(s, ""))
                continue
            keys.append("ESC"); i += 1; continue
        keys.append(CTRL.get(c, c if c >= " " else "")); i += 1
    return keys

def batch(keys):   # a typed Enter arrives alone; an Enter with more keys behind it is a pasted line break
    keys = [k for k in keys if k]
    return [("NL" if k == "ENTER" and i < len(keys) - 1 else k) for i, k in enumerate(keys)]

class ConsoleKeys:   # raw keys from the terminal: msvcrt on Windows (with VT input where the terminal has it), termios on POSIX
    WIN = {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT", "G": "HOME", "O": "END", "S": "DEL", "I": "PGUP", "Q": "PGDN", "s": "CTRL_LEFT", "t": "CTRL_RIGHT"}
    def __init__(self, mouse=True):
        self.win = os.name == "nt"; self.vt = False
        if self.win:
            if mouse and os.environ.get("WT_SESSION"):                   # Windows Terminal: VT input gives us the wheel and the same escape sequences as POSIX
                try:
                    import ctypes; k = ctypes.windll.kernel32; h = k.GetStdHandle(-10); m = ctypes.c_uint32()
                    if k.GetConsoleMode(h, ctypes.byref(m)): self.vt = bool(k.SetConsoleMode(h, (m.value | 0x0200) & ~0x0040)); self.old_mode = (h, m.value)
                except Exception: self.vt = False
        else:
            import termios, tty
            self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); tty.setcbreak(self.fd)
    def close(self):
        if self.win:
            if self.vt:
                try: import ctypes; ctypes.windll.kernel32.SetConsoleMode(*self.old_mode)
                except Exception: pass
        else:
            import termios; termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def poll(self, timeout):
        if self.win:
            import msvcrt
            t_end = time.time() + timeout; buf = []; keys = []
            while True:
                while msvcrt.kbhit():
                    c = msvcrt.getwch()
                    if c in ("\x00", "\xe0") and not self.vt: keys.append(self.WIN.get(msvcrt.getwch(), ""))
                    else: buf.append(c)
                if buf: keys += parse_escapes("".join(buf)); buf = []
                if keys or time.time() >= t_end: return batch(keys)
                time.sleep(0.01)
        import select
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r: return []
        data = os.read(self.fd, 4096).decode("utf-8", "replace")
        return batch(parse_escapes(data))

class ScriptedKeys:   # --keys: literal text plus <ENTER> <TAB> <UP> <DOWN> <PGUP> <PGDN> <BS> <ESC> <CTRLC> <CTRLK> <CTRLLEFT> <WHEELUP> <WHEELDOWN> <WAIT:s> <QUIT>, one key per poll
    NAMES = {"CTRLC": "CTRL_C", "CTRLK": "CTRL_K", "CTRLU": "CTRL_U", "CTRLW": "CTRL_W", "CTRLLEFT": "CTRL_LEFT", "CTRLRIGHT": "CTRL_RIGHT", "WHEELUP": "WHEEL_UP", "WHEELDOWN": "WHEEL_DOWN"}
    def __init__(self, script):
        self.items = re.findall(r"<[^>]+>|.", script, re.S); self.wait_until = 0
    def close(self): pass
    def poll(self, timeout):
        if time.time() < self.wait_until: time.sleep(min(timeout, self.wait_until - time.time())); return []
        if not self.items: time.sleep(timeout); return []
        k = self.items.pop(0)
        if k.startswith("<"):
            name = k[1:-1]
            if name.startswith("WAIT:"): self.wait_until = time.time() + float(name[5:]); return []
            return [self.NAMES.get(name, name)]
        return [k]
