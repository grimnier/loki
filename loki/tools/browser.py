# tools/browser.py -- a real browser for the model: Edge or Chrome, headless, driven over the DevTools protocol with a
#   standard-library websocket. Pages that need JavaScript, clicking, typing, scrolling, screenshots. One tab is kept
#   open and reused; `browse` loads a page and reads it, the others act on the page that is up.
#   The browser is found in this order: LOKI_BROWSER (env or "keys" in settings), Edge / Chrome on Windows, then
#   microsoft-edge / google-chrome / chromium on the path. Its profile lives in results/browser_profile.
import os, re, json, time, base64, socket, struct, subprocess, shutil, urllib.request, threading
from . import tool, HERE
from .web import window, key

def find_browser():
    env = key("LOKI_BROWSER")
    if env and os.path.isfile(env): return env
    cands = [os.path.join(os.environ.get(v, ""), *p) for v in ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData") for p in (("Microsoft", "Edge", "Application", "msedge.exe"), ("Google", "Chrome", "Application", "chrome.exe"))]
    cands += [shutil.which(n) or "" for n in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser")] + ["/opt/pw-browsers/chromium"]
    return next((c for c in cands if c and os.path.isfile(c)), None)

class WS:   # the little of RFC 6455 a DevTools client needs: text frames, masked out, unmasked in, ping answered
    def __init__(self, url):
        m = re.match(r"ws://([^:/]+):(\d+)(/.*)", url); host, port, path = m.group(1), int(m.group(2)), m.group(3)
        self.s = socket.create_connection((host, port), timeout=30)
        self.s.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {base64.b64encode(os.urandom(16)).decode()}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
        buf = b""
        while b"\r\n\r\n" not in buf: buf += self.s.recv(4096)
        if b" 101 " not in buf.split(b"\r\n", 1)[0]: raise RuntimeError("websocket handshake refused")
        self.rest = buf.split(b"\r\n\r\n", 1)[1]; self.lock = threading.Lock()
    def send(self, text):
        data = text.encode("utf-8"); mask = os.urandom(4); n = len(data)
        head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536 else bytes([0x80 | 127]) + struct.pack(">Q", n))
        with self.lock: self.s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
    def _read(self, n):
        while len(self.rest) < n:
            chunk = self.s.recv(65536)
            if not chunk: raise RuntimeError("browser closed the connection")
            self.rest += chunk
        out, self.rest = self.rest[:n], self.rest[n:]; return out
    def recv(self):
        while True:
            b1, b2 = self._read(2); op = b1 & 0x0F; n = b2 & 0x7F
            if n == 126: n = struct.unpack(">H", self._read(2))[0]
            elif n == 127: n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b2 & 0x80 else None; data = self._read(n)
            if mask: data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if op == 0x9: self.s.sendall(bytes([0x8A, 0x80]) + os.urandom(4) + b""); continue   # ping → pong (empty, masked)
            if op == 0x8: raise RuntimeError("browser closed the connection")
            if op in (0x1, 0x0): return data.decode("utf-8", "replace")
    def close(self):
        try: self.s.close()
        except Exception: pass

class Browser:
    _one = None
    def __init__(self):
        exe = find_browser()
        if not exe: raise RuntimeError("no Edge, Chrome or Chromium on this machine (set LOKI_BROWSER to the executable)")
        prof = os.path.join(HERE, "results", "browser_profile"); os.makedirs(prof, exist_ok=True)
        port_file = os.path.join(prof, "DevToolsActivePort")
        if os.path.exists(port_file): os.remove(port_file)
        self.p = subprocess.Popen([exe, "--headless=new", "--remote-debugging-port=0", "--user-data-dir=" + prof, "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--no-sandbox",
                                   "--disable-background-networking", "--disable-component-update", "--disable-sync", "--window-size=1280,900", "--hide-scrollbars", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if os.path.exists(port_file) and open(port_file).read().strip(): break
            time.sleep(0.1)
        else: raise RuntimeError("the browser did not start")
        self.port = int(open(port_file).read().split()[0]); self.ws = None; self.id = 0; self.events = []
        self.open_tab()
    def http(self, path, method="GET"):
        with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method), timeout=10) as r: return json.loads(r.read().decode())
    def open_tab(self):
        try: tab = self.http("/json/new?about:blank", "PUT")
        except Exception: tab = self.http("/json/new?about:blank")
        self.ws = WS(tab["webSocketDebuggerUrl"]); self.call("Page.enable"); self.call("Runtime.enable")
    def call(self, method, timeout=30, **params):
        self.id += 1; mid = self.id; self.ws.send(json.dumps({"id": mid, "method": method, "params": params})); t0 = time.time()
        while time.time() - t0 < timeout:
            m = json.loads(self.ws.recv())
            if m.get("id") == mid:
                if "error" in m: raise RuntimeError(m["error"].get("message", "browser error"))
                return m.get("result", {})
            if "method" in m: self.events.append(m["method"])
        raise RuntimeError(f"{method} timed out")
    def wait_load(self, timeout=20, grace=1.2):   # until the page has loaded; or, if nothing even starts loading within grace seconds, straight back (a click that only ran a script)
        t0 = time.time(); started = False
        while time.time() - t0 < timeout:
            if "Page.loadEventFired" in self.events: self.events.clear(); time.sleep(0.4); return True
            started = started or any(e in ("Page.frameStartedLoading", "Page.frameNavigated") for e in self.events)
            if not started and time.time() - t0 > grace: time.sleep(0.2); return False
            try:
                self.ws.s.settimeout(0.3); m = json.loads(self.ws.recv())
                if m.get("method"): self.events.append(m["method"])
            except (socket.timeout, TimeoutError): pass
            finally: self.ws.s.settimeout(30)
        return False
    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if r.get("exceptionDetails"): raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", "script error")[:200])
        return r.get("result", {}).get("value")
    def alive(self): return self.p.poll() is None and self.ws is not None
    @classmethod
    def get(cls):
        if cls._one is None or not cls._one.alive():
            if cls._one: cls._one.quit()
            cls._one = Browser()
        return cls._one
    def quit(self):
        try: self.ws and self.ws.close(); self.p.kill()
        except Exception: pass

READ = """(() => { const t = (document.body ? document.body.innerText : "").replace(/\\n{3,}/g, "\\n\\n").trim(); return JSON.stringify({title: document.title, url: location.href, text: t}); })()"""
LINKS = """(() => JSON.stringify(Array.from(document.querySelectorAll("a[href]")).map(a => [a.innerText.trim().slice(0, 80), a.href]).filter(x => x[1].startsWith("http"))))()"""
FIND = """(sel) => { let el = null; try { el = document.querySelector(sel); } catch (e) {}
  if (!el) { const want = sel.toLowerCase(); for (const e of document.querySelectorAll("a, button, input, [role=button], label, summary, option")) { const t = (e.innerText || e.value || e.getAttribute("aria-label") || "").trim().toLowerCase(); if (t === want || (t && t.includes(want) && t.length < want.length + 40)) { el = e; break; } } }
  return el; }"""

def page_text(b, start=0, find=None):
    j = json.loads(b.js(READ)); head = f"{j['title']}\n{j['url']}\n" if j["title"] else f"{j['url']}\n"
    text = j["text"]
    if find:
        hits = [m.start() for m in re.finditer(re.escape(str(find)), text, re.I)]
        if not hits: return head + f'"{find}" is not on this page ({len(text)} characters)'
        parts = []; last = -1
        for h in hits[:8]:
            a = max(0, text.rfind("\n", 0, max(0, h - 300))); e = text.find("\n", h + 300); e = len(text) if e < 0 else e
            if a <= last: continue
            parts.append(f"[at {a}]\n" + text[a:e].strip()); last = e
        return head + f'{len(hits)} match{"es" if len(hits) != 1 else ""} for "{find}"\n\n' + "\n…\n".join(parts)
    return head + window(text, start, 5000)

@tool("browse", "open a page in a real browser (javascript runs) and read it as text; long pages come in parts with start=, find= shows the lines around a word; no url = the page that is up",
      {"url": ("the page", False), "start": ("character offset to continue from", False), "find": ("a word or phrase to look for", False)}, cap=6000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def browse(url=None, start=0, find=None):
    b = Browser.get()
    if url:
        if not re.match(r"^[a-z]+://", url): url = "https://" + url
        b.events.clear(); b.call("Page.navigate", url=url); b.wait_load()
    return page_text(b, start, find)

@tool("click", "click something on the page that is up: a css selector, or the visible text of a link or button; then read the page",
      {"target": ("css selector or visible text", True)}, cap=6000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def click(target):
    b = Browser.get(); b.events.clear()
    ok = b.js(f"(() => {{ const el = ({FIND})({json.dumps(str(target))}); if (!el) return false; el.scrollIntoView({{block: 'center'}}); el.click(); return true; }})()")
    if not ok: return f"nothing on the page matches {target}"
    b.wait_load(timeout=8); return page_text(b)

@tool("type", "type into a field on the page that is up (css selector, placeholder or label text), optionally press Enter; then read the page",
      {"target": ("css selector, placeholder or label text", True), "text": ("what to type", True), "enter": ("true to press Enter after", False)}, cap=6000, policy="ask", summary=lambda a, t: t.split("\n", 1)[0][:60])
def type_(target, text, enter=False):
    b = Browser.get(); sel = json.dumps(str(target))
    ok = b.js(f"""(() => {{ let el = ({FIND})({sel}); if (!el) {{ for (const e of document.querySelectorAll("input, textarea, [contenteditable]")) {{ const t = ((e.placeholder || "") + " " + (e.getAttribute("aria-label") || "") + " " + (e.name || "")).toLowerCase(); if (t.includes({sel}.toLowerCase())) {{ el = e; break; }} }} }}
      if (!el) return false; el.scrollIntoView({{block: 'center'}}); el.focus(); if ("value" in el) el.value = ""; return true; }})()""")
    if not ok: return f"no field on the page matches {target}"
    b.call("Input.insertText", text=str(text))
    if enter and str(enter).lower() != "false":
        b.events.clear()
        for t in ("keyDown", "keyUp"): b.call("Input.dispatchKeyEvent", type=t, key="Enter", code="Enter", windowsVirtualKeyCode=13, nativeVirtualKeyCode=13, text="\r" if t == "keyDown" else "")
        b.wait_load(timeout=10)
    return page_text(b)

@tool("scroll", "scroll the page that is up: down, up, top or bottom; then read what is in view", {"where": ("down, up, top or bottom; default down", False)}, cap=6000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def scroll(where="down"):
    b = Browser.get(); w = str(where or "down").lower()
    b.js({"up": "window.scrollBy(0, -window.innerHeight * 0.9)", "top": "window.scrollTo(0, 0)", "bottom": "window.scrollTo(0, document.body.scrollHeight)"}.get(w, "window.scrollBy(0, window.innerHeight * 0.9)")); time.sleep(0.3)
    y = b.js("window.scrollY"); h = b.js("document.body.scrollHeight - window.innerHeight") or 1
    start = int(len(json.loads(b.js(READ))["text"]) * min(1.0, max(0.0, y / h))) if h else 0
    return page_text(b, start=max(0, start - 500))

@tool("screenshot", "a picture of the page that is up, saved into results/browser", {"name": ("file name, default a timestamp", False)}, cap=300, summary=lambda a, t: t)
def screenshot(name=None):
    b = Browser.get(); d = os.path.join(HERE, "results", "browser"); os.makedirs(d, exist_ok=True)
    name = re.sub(r"[^\w.\-]+", "_", str(name or time.strftime("%Y%m%d-%H%M%S")))[:80]; name = name if name.endswith(".png") else name + ".png"
    data = base64.b64decode(b.call("Page.captureScreenshot", format="png")["data"]); open(os.path.join(d, name), "wb").write(data)
    return f"saved results/browser/{name} ({len(data) // 1024} KB)"

@tool("browse_links", "the links on the page that is up, numbered, so one can be followed with browse or click", {"filter": ("only links whose text or url contains this", False), "start": ("skip this many", False)}, cap=3000, summary=lambda a, t: f"{t.count(chr(10)) + 1} links")
def browse_links(filter=None, start=0):
    rows = json.loads(Browser.get().js(LINKS)); seen = set(); rows = [r for r in rows if not (r[1] in seen or seen.add(r[1]))]
    if filter: rows = [r for r in rows if str(filter).lower() in (r[0] + r[1]).lower()]
    start = int(start or 0); shown = rows[start:start + 40]
    if not shown: return "no links" + (f' matching "{filter}"' if filter else "")
    return "\n".join(f"{start + i + 1}. {t or '(no text)'} — {u}" for i, (t, u) in enumerate(shown)) + (f"\n[more: start={start + 40}]" if len(rows) > start + 40 else "")
