# app.py -- loki as a window: the same client as chat.py, drawn by a browser instead of a terminal. Standard library only.
#   Starts a local server (one session, one engine), opens the page as a chromeless app window (Edge or Chrome
#   --app=, else the default browser) and streams the conversation to it over server-sent events. The engine
#   protocol, the turn logic, settings and transcripts are chat.py's; this file only adds the HTTP side.
#   app.py [--replay [tok/s]] [--port N] [--no-open]
import sys, os, json, time, queue, threading, subprocess, shutil, webbrowser, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import chat, tools
from chat import TUI, Quit, load_settings, Replay, Engine

LOKI = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(LOKI); WEB = os.path.join(LOKI, "web")   # the page lives in loki/web, results in the repo

class Session(TUI):   # chat.py's turn logic with the screen replaced by a state snapshot that the page renders
    def __init__(self, cfg, eng):
        super().__init__(cfg, eng, None, None)
        self.cmds = queue.Queue(); self.clients = []; self.seq = 0; self.lock = threading.Lock()
        self.last_ents = []; self.last_rest = None; self.full = None   # what each client already has, for deltas; full: the snapshot a new client starts from
    def state(self):
        t = self.turn
        turn = None if not t else {"think": t["think"][:t["think_shown"]], "t0": t["t0"], "in_think": t["in_think"] or bool(t.get("hood_close")), "stopped": t["stopped"], "stage": t["stage"], "hop": t["hop"],
                                   "tool": {"text": tools.describe(*t["tool"][:2]), "t0": t["tool"][2]} if t.get("tool") else None,
                                   "pending": {"name": t["pending"][0], "text": tools.describe(*t["pending"])} if t.get("pending") else None}
        return {"log": self.log, "turn": turn, "ready": self.eng.ready, "pos": self.eng.pos, "ctx": self.eng.ctx, "dead": self.dead, "t_start": self.t_start,
                "conv": self.conv and {"id": self.conv["id"], "title": self.conv["title"]}, "queue": list(self.queue), "branches": len(self.branches), "done_at": self.done_at,
                "cmds": [{"name": n, "args": a, "desc": d} for n, a, d in chat.COMMANDS],
                "cfg": {k: self.cfg.get(k) for k in ("think", "answer", "bars", "stats", "ctx", "tools", "hops", "tool_cap")},
                "tools": {"enabled": tools.enabled(self.cfg), "list": [{"name": n, "desc": tools.REGISTRY[n].desc, "policy": tools.policy(n, self.cfg)} for n in tools.REGISTRY],
                          "rules": self.cfg.get("tool_rules") or []}}
    def render(self):                                                 # called wherever the TUI would draw; publishes a delta when something changed
        s = self.state(); ents = [json.dumps(e, ensure_ascii=False) for e in s.pop("log")]; rest = json.dumps(s, ensure_ascii=False)
        common = 0
        while common < min(len(ents), len(self.last_ents)) and ents[common] == self.last_ents[common]: common += 1
        if common == len(ents) == len(self.last_ents) and rest == self.last_rest: return
        self.seq += 1; s["seq"] = self.seq
        delta = json.dumps({**s, "log_from": common, "log": [json.loads(e) for e in ents[common:]]}, ensure_ascii=False)   # entries from the first changed one on
        self.full = json.dumps({**s, "full": True, "log_from": 0, "log": [json.loads(e) for e in ents]}, ensure_ascii=False)
        self.last_ents, self.last_rest = ents, rest
        with self.lock:
            for q in self.clients: q.put(delta)
    def subscribe(self):
        q = queue.Queue()
        with self.lock: self.clients.append(q)
        if self.full: q.put(self.full)
        return q
    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients: self.clients.remove(q)
    def loop(self):
        try:
            while True:
                try:
                    while True:
                        kind, arg = self.cmds.get_nowait()
                        if kind == "send":                                # as the terminal's Enter: a command runs now, a message during a turn waits its turn
                            if self.turn and not arg.startswith("/"): self.queue.append(arg)
                            else: self.set_box(arg); self.submit()
                        elif kind == "stop": self.handle("CTRL_C") if self.turn else None
                        elif kind == "new": self.command("/new")
                        elif kind == "open": self.open_conv(arg) if not self.turn else None
                        elif kind == "tool": self.answer_tool(arg in ("allow", "always"), always=arg == "always")
                        elif kind == "queue": self.queue = []
                        elif kind == "edit":                              # arg: (n, box) — the message goes back to the page's composer, the old tail is kept as a branch
                            n, box = arg; users = [i for i, e in enumerate(self.log) if e["kind"] == "user"]
                            if not self.turn and 0 < n <= len(users): i = users[-n]; box["text"] = self.log[i]["text"]; self.branch_at(i)
                            box["done"].set()
                except queue.Empty: pass
                self.pump(); self.render(); time.sleep(0.05)
        except Quit: pass
        finally: self.quit(); os._exit(0)

class Handler(BaseHTTPRequestHandler):
    session = None; runner = None
    def log_message(self, *a): pass
    def send_file(self, name):
        path = os.path.realpath(os.path.join(WEB, name))
        if not path.startswith(os.path.realpath(WEB) + os.sep) or not os.path.isfile(path): self.send_error(404); return   # only app/ is served
        ctype = {"html": "text/html", "css": "text/css", "js": "text/javascript", "svg": "image/svg+xml"}.get(name.rsplit(".", 1)[-1], "application/octet-stream")
        data = open(path, "rb").read(); self.send_response(200); self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p == "/": return self.send_file("index.html")
        if p == "/events": return self.events()
        if p == "/state": return self.reply(self.session.state())
        if p == "/favicon.ico": self.send_response(204); self.end_headers(); return
        if p == "/studio": return self.reply(tools.local.load_queue())
        if p.startswith("/out/"):                                     # finished renders, from results/studio/out only
            name = os.path.basename(p[5:]); path = os.path.join(HERE, "results", "studio", "out", name)
            if not os.path.isfile(path): self.send_error(404); return
            data = open(path, "rb").read(); self.send_response(200); self.send_header("Content-Type", {"png": "image/png", "mp4": "video/mp4", "txt": "text/plain"}.get(name.rsplit(".", 1)[-1], "application/octet-stream"))
            self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "max-age=3600"); self.end_headers(); self.wfile.write(data); return
        if p == "/conversations": return self.reply(TUI.list_convs(200))
        if p == "/find": return self.reply(TUI.find_convs(urllib.parse.unquote(self.path.partition("?q=")[2]), 50))
        return self.send_file(p.lstrip("/"))
    def do_POST(self):
        try: n = int(self.headers.get("Content-Length") or 0); body = json.loads(self.rfile.read(min(n, 1 << 20)) or b"{}"); assert isinstance(body, dict)
        except Exception: self.send_error(400); return
        if self.path == "/send": self.session.cmds.put(("send", str(body.get("text", ""))))
        elif self.path == "/stop": self.session.cmds.put(("stop", None))
        elif self.path == "/new": self.session.cmds.put(("new", None))
        elif self.path == "/open": self.session.cmds.put(("open", str(body.get("id", ""))))
        elif self.path == "/delete":                                  # a conversation file, by id
            cid = str(body.get("id", "")); path = os.path.join(chat.CONVS, cid + ".json")
            if cid.replace("-", "").isdigit() and os.path.isfile(path): os.remove(path)
        elif self.path == "/tool": self.session.cmds.put(("tool", str(body.get("answer", "allow" if body.get("allow") else "deny"))))   # allow / deny / always when a tool asks first
        elif self.path == "/queue": self.session.cmds.put(("queue", None))                                      # clear what is waiting
        elif self.path == "/edit":                                    # the Nth-last message back into the composer; the old tail becomes a branch
            box = {"text": None, "done": threading.Event()}; self.session.cmds.put(("edit", (int(body.get("n", 1)), box))); box["done"].wait(3)
            self.reply({"text": box["text"]}); return
        elif self.path == "/studio":                                  # the studio queue is results/studio/queue.json, shared with the studio tool
            q = tools.local.load_queue()
            if body.get("drop") is not None:
                q = [j for j in q if str(j.get("id")) != str(body["drop"])]
                if Handler.runner: Handler.runner.cancel(body["drop"])
            elif body.get("job"): j = dict(body["job"]); j.setdefault("id", int(time.time() * 1000)); j.setdefault("status", "waiting for the model"); j["by"] = "you"; q.append(j)
            tools.local.save_queue(q); self.reply(q); return
        else: self.send_error(404); return
        self.reply({"ok": True})
    def reply(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8"); self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def events(self):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-store"); self.end_headers()
        q = self.session.subscribe()
        try:
            while True:
                try: s = q.get(timeout=15); self.wfile.write(f"data: {s}\n\n".encode("utf-8"))
                except queue.Empty: self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError): pass
        finally: self.session.unsubscribe(q)

def open_window(url):   # a chromeless app window where Edge or Chrome is around, else a tab
    cands = [os.path.join(os.environ.get(v, ""), *p) for v in ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData") for p in (("Microsoft", "Edge", "Application", "msedge.exe"), ("Google", "Chrome", "Application", "chrome.exe"))]
    cands += [shutil.which(n) or "" for n in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser")]
    for exe in cands:
        if exe and os.path.isfile(exe):
            try: subprocess.Popen([exe, f"--app={url}", "--window-size=1120,780", "--user-data-dir=" + os.path.join(HERE, "results", "app_profile")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return
            except Exception: pass
    webbrowser.open(url)

if __name__ == "__main__":
    chat.console_setup(); argv = sys.argv[1:]
    def opt(name, default=None, conv=str):
        if name not in argv: return None
        i = argv.index(name); return conv(argv[i + 1]) if i + 1 < len(argv) and not argv[i + 1].startswith("--") else default
    replay = opt("--replay", 4.0, float); port = opt("--port", 8765, int) or 8765
    cfg = load_settings()
    if "--sandbox" in argv: cfg["memory"] = "sandbox"               # this run only, as in chat.py; /sandbox inside the chat saves the choice
    if "--shared" in argv: cfg["memory"] = "shared"
    Engine.sweep_sandbox(); eng = Replay(cfg, replay) if replay else Engine(cfg)
    Handler.session = s = Session(cfg, eng); threading.Thread(target=s.loop, daemon=True).start()
    if cfg.get("studio_runner", True):                                # the studio queue is worked by the agent in studio/ (the stub until the 4K model is tied in)
        from studio.runner import Runner; Handler.runner = Runner(cfg); Handler.runner.start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler); srv.daemon_threads = True
    url = f"http://127.0.0.1:{srv.server_address[1]}/"; print(url, flush=True)
    if "--no-open" not in argv: open_window(url)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
