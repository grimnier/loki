# tools -- the tool belt the client holds for the model. The model only ever emits text; when its answer carries a call
#   like <tool>{"name": "search", "args": {"q": "rtx 4050 vram"}}</tool> the client runs the tool here, shows the model the
#   result as the next user turn, and the model carries on. Nothing in here touches the engine; chat.py owns the loop.
#   A tool is a plain function registered with @tool: a name, a one-line description, its parameters, a result cap and a
#   policy (allow: runs; ask: the owner confirms first; off: not offered). Policies and caps are overridable from
#   chat_settings.json ("tool_policy": {"shell": "off"}, "tool_cap": 4000). Every run is appended to results/tools/<date>.jsonl.
import json, os, re, time, datetime, threading

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo: results/, chat_settings.json
REGISTRY = {}

def timeout(name, cfg):   # seconds a tool may take; tool_timeouts {"fetch": 60} per tool, tool_timeout for all
    return int((cfg.get("tool_timeouts") or {}).get(name, cfg.get("tool_timeout") or 90))

class Tool:
    def __init__(self, name, desc, params, fn, cap, policy, summary):
        self.name, self.desc, self.params, self.fn, self.cap, self.policy, self.summary = name, desc, params, fn, cap, policy, summary
    def spec(self):   # one line the model reads: name(arg: what, arg?: what) — description
        args = ", ".join(f"{k}{'' if v[1] else '?'}: {v[0]}" for k, v in self.params.items())
        return f"{self.name}({args}) — {self.desc}"

def tool(name, desc, params=None, cap=4000, policy="allow", summary=None):
    """params: {"arg": ("what it is", required: bool)}. cap: characters of result the model sees (about a quarter in tokens).
    summary: fn(args, result) -> the short tail of the dim line in the log, e.g. '3 results'."""
    def wrap(fn):
        REGISTRY[name] = Tool(name, desc, params or {}, fn, cap, policy, summary); return fn
    return wrap

class Result:
    def __init__(self, name, args, text, ok=True, secs=0.0, summary=""):
        self.name, self.args, self.text, self.ok, self.secs, self.summary = name, args, text, ok, secs, summary

# ---- what the model is told. Short on purpose: it sits in every fresh context.
def preamble(cfg):
    names = offered(cfg)
    if not names: return ""
    lines = [REGISTRY[n].spec() for n in names]
    return ("You have tools. To use one, end your answer with exactly one call on its own line:\n"
            '<tool>{"name": "search", "args": {"q": "..."}}</tool>\n'
            "The result comes back in the next user message as <result name=\"...\">...</result>; then continue, calling another "
            "tool if needed or giving the final answer. Call a tool when it would make the answer more correct or current; "
            "do not call one when you already know. Tools:\n" + "\n".join(lines) + "\n")

def enabled(cfg):   # "auto" (the default) offers tools only when the context has room for the list (about 1100 tokens): 2048 positions or more
    v = cfg.get("tools", "auto")
    return v is True or (v == "auto" and int(cfg.get("ctx", 0)) >= 2048)

def offered(cfg):
    return [n for n in REGISTRY if policy(n, cfg) != "off"] if enabled(cfg) else []

def policy(name, cfg):
    return (cfg.get("tool_policy") or {}).get(name, REGISTRY[name].policy)

def cap(name, cfg):
    return int((cfg.get("tool_caps") or {}).get(name, cfg.get("tool_cap") or REGISTRY[name].cap))

# ---- remembered answers: "always allow shell ls", kept in settings as tool_rules [{"tool": "shell", "match": "ls", "policy": "allow"}]
def decide(name, args, cfg):
    """allow / ask / off for this call: a rule whose match is a prefix of the call's text wins, else the tool's policy."""
    text = call_text(name, args)
    for r in cfg.get("tool_rules") or []:
        if r.get("tool") == name and text.startswith(str(r.get("match", ""))): return r.get("policy", "allow")
    return policy(name, cfg)

def call_text(name, args):   # the part of a call a rule matches on: the main argument, e.g. "ls -la" for shell
    main = next((str(args[k]) for k in ("cmd", "code", "url", "q", "path", "expr", "place", "pattern", "prompt", "text") if k in args), "")
    return " ".join(main.split())

def remember(name, args, cfg, policy="allow", whole=False):
    """Add a rule for this call: the first word of its argument (shell ls, python …) unless whole, then the exact text."""
    text = call_text(name, args); match = text if whole or not text else text.split(" ", 1)[0]
    rules = [r for r in cfg.get("tool_rules") or [] if not (r.get("tool") == name and r.get("match") == match)]
    rules.append({"tool": name, "match": match, "policy": policy}); cfg["tool_rules"] = rules
    return f"{name} {match}".strip()

# ---- finding a call in the answer. The tag form is canonical; a ```json block with name/args is accepted too.
CALL = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.S)
FENCE = re.compile(r"```(?:json|tool)?\s*(\{[^`]*?\"name\"[^`]*?\})\s*```", re.S)
def parse_call(answer):
    """Returns (name, args, answer_without_the_call) or None. Only the last call counts; the model is told one per turn."""
    for rx in (CALL, FENCE):
        ms = list(rx.finditer(answer))
        if not ms: continue
        m = ms[-1]
        try: obj = json.loads(m.group(1))
        except Exception: continue
        name = str(obj.get("name", "")).strip(); args = obj.get("args") or {}
        if not isinstance(args, dict): args = {"q": str(args)}
        if name: return name, args, (answer[:m.start()] + answer[m.end():]).strip()
    return None

def describe(name, args):   # the dim line's head: → search "rtx 4050 vram"
    short = lambda v: (s := " ".join(str(v).split()))[:80] + ("…" if len(s) > 80 else "")
    a = ", ".join(short(v) if k in ("q", "url", "expr", "cmd", "path", "code", "text", "prompt", "pattern", "place") else f"{k}={short(v)}" for k, v in args.items() if v not in ("", None))
    return f"→ {name} {a}" if a else f"→ {name}"

def run(name, args, cfg):
    t = REGISTRY.get(name)
    if not t or policy(name, cfg) == "off": return Result(name, args, f"no such tool: {name}. Tools: {', '.join(offered(cfg))}", ok=False)
    missing = [k for k, v in t.params.items() if v[1] and k not in args]
    if missing: return Result(name, args, f"{name} needs {', '.join(missing)}", ok=False)
    t0 = time.time(); box = {}
    def go():
        try: box["text"] = t.fn(**{k: v for k, v in args.items() if k in t.params}); box["ok"] = True
        except Exception as e: box["text"] = f"{name} failed: {e}"; box["ok"] = False
    th = threading.Thread(target=go, daemon=True); th.start(); th.join(timeout(name, cfg))   # a tool that hangs is abandoned, not waited on
    if th.is_alive(): text, ok = f"{name} gave up after {timeout(name, cfg)}s", False
    else: text, ok = box["text"], box["ok"]
    secs = time.time() - t0; text = str(text); limit = cap(name, cfg)
    if len(text) > limit: text = text[:limit].rstrip() + f"\n[cut at {limit} characters of {len(text)}]"
    summary = (t.summary(args, text) if t.summary and ok else "") or ("" if ok else "failed")
    r = Result(name, args, text, ok, secs, summary); log(r); return r

def log(r):
    try:
        os.makedirs(os.path.join(HERE, "results", "tools"), exist_ok=True)
        with open(os.path.join(HERE, "results", "tools", datetime.date.today().isoformat() + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "name": r.name, "args": r.args, "ok": r.ok, "secs": round(r.secs, 2), "chars": len(r.text), "text": r.text[:2000]}, ensure_ascii=False) + "\n")
    except Exception: pass

def wrap_result(r):   # what the model reads back
    return f'<result name="{r.name}"{"" if r.ok else " error=\"true\""}>\n{r.text}\n</result>'

from . import local, system, web, browser   # noqa: E402  (registering)
