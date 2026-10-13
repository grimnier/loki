# tools/local.py -- the machine the client runs on: the clock, a calculator, the model's own notes, files, a shell and
#   python (both ask first), and the studio queue, which is how the model hands a picture or a video to the 4K agent.
import os, ast, math, json, time, datetime, subprocess, sys, shutil
from . import tool, HERE

@tool("clock", "the date and time now", {}, cap=200)
def clock():
    n = datetime.datetime.now().astimezone(); return n.strftime("%A %Y-%m-%d %H:%M %Z") + f" (unix {int(n.timestamp())})"

SAFE = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}; SAFE.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len, "int": int, "float": float, "pow": pow})
@tool("calc", "evaluate a math expression (python syntax, math functions; ** is power)", {"expr": ("the expression", True)}, cap=500)
def calc(expr):
    expr = str(expr).replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    for n in ast.walk(tree):
        if isinstance(n, (ast.Attribute, ast.Import, ast.Lambda, ast.Subscript)) or (isinstance(n, ast.Name) and n.id not in SAFE) or (isinstance(n, ast.Call) and not isinstance(n.func, ast.Name)):
            raise ValueError("only numbers, operators and math functions")
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) and not (isinstance(n.right, ast.Constant) and isinstance(n.right.value, (int, float)) and abs(n.right.value) <= 10000):
            raise ValueError("an exponent must be a plain number up to 10000")   # 9**9**9 would never return
    v = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, SAFE)
    return f"{expr} = {v:,}" if isinstance(v, int) and abs(v) >= 10000 else f"{expr} = {v}"

@tool("clipboard", "what is on the clipboard right now", {}, cap=4000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def clipboard():
    for cmd in (["powershell", "-NoProfile", "-Command", "Get-Clipboard"], ["wl-paste"], ["xclip", "-selection", "clipboard", "-o"], ["pbpaste"]):
        if shutil.which(cmd[0]):
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0: return r.stdout.strip() or "(the clipboard is empty)"
    return "no clipboard tool on this machine"

NOTES = os.path.join(HERE, "results", "notes.md")
@tool("notes", "the model's own notebook across conversations: read it, add a line, or search it", {"action": ("read, add or find", True), "text": ("what to add, or what to find", False)}, cap=3000, summary=lambda a, t: a.get("action", ""))
def notes(action, text=None):
    os.makedirs(os.path.dirname(NOTES), exist_ok=True); body = open(NOTES, encoding="utf-8").read() if os.path.isfile(NOTES) else ""
    if action == "add":
        if not text: return "nothing to add"
        with open(NOTES, "a", encoding="utf-8") as f: f.write(f"- {datetime.date.today().isoformat()} · {str(text).strip()}\n")
        return "noted"
    if action == "find":
        hits = [l for l in body.splitlines() if text and str(text).lower() in l.lower()]; return "\n".join(hits) or f"nothing about {text}"
    return body[-3000:] or "the notebook is empty"

def safe_path(path):   # inside the repo, or an absolute path the owner gave
    p = os.path.abspath(os.path.join(HERE, os.path.expanduser(str(path)))); return p

@tool("files", "list a folder or read a file on this machine", {"path": ("file or folder, relative to the repo or absolute", True), "start": ("line to start reading from", False)}, cap=6000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def files(path, start=0):
    p = safe_path(path)
    if os.path.isdir(p):
        rows = sorted(os.listdir(p)); return f"{len(rows)} entries in {path}\n" + "\n".join((r + "/" if os.path.isdir(os.path.join(p, r)) else f"{r} · {os.path.getsize(os.path.join(p, r))} B") for r in rows[:200])
    if not os.path.isfile(p): return f"no such file: {path}"
    if os.path.getsize(p) > 20 * 1024 * 1024: return "too large to read"
    lines = open(p, encoding="utf-8", errors="replace").read().splitlines(); start = int(start or 0)
    chunk = lines[start:start + 200]
    return f"{path} · {len(lines)} lines" + (f" · from {start}" if start else "") + "\n" + "\n".join(f"{start + i + 1}: {l}" for i, l in enumerate(chunk)) + (f"\n[more: start={start + 200}]" if len(lines) > start + 200 else "")

@tool("write", "write text to a file (creates or replaces it)", {"path": ("the file", True), "text": ("the whole content", True)}, cap=200, policy="ask")
def write(path, text):
    p = safe_path(path); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w", encoding="utf-8").write(str(text)); return f"wrote {len(str(text))} characters to {path}"

@tool("shell", "run a command on this machine and return its output", {"cmd": ("the command", True)}, cap=6000, policy="ask", summary=lambda a, t: t.split("\n", 1)[0][:60])
def shell(cmd):
    r = subprocess.run(str(cmd), shell=True, capture_output=True, text=True, timeout=120, cwd=HERE)
    out = (r.stdout + ("\n" + r.stderr if r.stderr.strip() else "")).strip(); return (out or "(no output)") + (f"\n[exit {r.returncode}]" if r.returncode else "")

@tool("python", "run a python snippet and return what it prints", {"code": ("the code", True)}, cap=6000, policy="ask", summary=lambda a, t: t.split("\n", 1)[0][:60])
def python(code):
    r = subprocess.run([sys.executable, "-I", "-c", str(code)], capture_output=True, text=True, timeout=120, cwd=HERE)
    out = (r.stdout + ("\n" + r.stderr if r.stderr.strip() else "")).strip(); return (out or "(no output)") + (f"\n[exit {r.returncode}]" if r.returncode else "")

# ---- the studio queue: results/studio/queue.json, shared with the window's studio tab and, later, the 4K agent
QUEUE = os.path.join(HERE, "results", "studio", "queue.json")
def load_queue():
    try: return json.load(open(QUEUE, encoding="utf-8"))
    except Exception: return []
def save_queue(q):
    os.makedirs(os.path.dirname(QUEUE), exist_ok=True); json.dump(q, open(QUEUE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

@tool("studio", "hand a picture or a video to the 4K studio: queue a job, list the queue, or drop one",
      {"action": ("render, list or drop", True), "prompt": ("what to make", False), "kind": ("picture or video, default picture", False), "ratio": ("16:9, 1:1 or 9:16", False), "size": ("4K or 1080p", False), "length": ("seconds, video only", False), "id": ("job id to drop", False)},
      cap=1500, summary=lambda a, t: t.split("\n", 1)[0][:60])
def studio(action, prompt=None, kind="picture", ratio="16:9", size="4K", length=None, id=None):
    q = load_queue()
    if action == "render":
        if not prompt: return "a job needs a prompt"
        w, h = (3840, 2160) if size == "4K" else (1920, 1080); px = f"{w} × {h}" if ratio == "16:9" else f"{h} × {w}" if ratio == "9:16" else f"{h} × {h}"
        job = {"id": int(time.time() * 1000), "prompt": str(prompt), "kind": kind if kind in ("picture", "video") else "picture", "size": px, "dur": f"{length or 4}s" if kind == "video" else "",
               "steps": "30", "guide": "6", "seed": "random", "status": "waiting for the model", "by": "model"}
        q.append(job); save_queue(q); return f"queued #{len(q)} · {job['kind']} · {px} · {job['prompt'][:60]}"
    if action == "drop":
        n = len(q); q = [j for j in q if str(j.get("id")) != str(id)]; save_queue(q); return "dropped" if len(q) < n else f"no job {id}"
    return "\n".join(f"#{i + 1} {j['kind']} · {j['size']} · {j['status']} · {j['prompt'][:70]}" for i, j in enumerate(q)) or "the queue is empty"
