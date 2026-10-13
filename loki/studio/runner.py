# studio/runner.py -- turns the studio queue (results/studio/queue.json) over to an agent, one job at a time, and keeps the
#   queue honest about progress. The agent is any program that speaks three lines over stdin/stdout:
#       runner → agent:  JOB <json>                 one job: id, prompt, kind (picture|video), size, dur, steps, guide, seed, out (a path to write)
#       agent → runner:  PROG <0-100> [note]        as often as it likes
#                        DONE <path> [note]         the finished file (under results/studio/out)
#                        FAIL <reason>
#   The default agent is studio/stub_agent.py, which draws a placeholder so the whole path works today; the 4K model
#   replaces it with one setting ("studio_agent": ["python", "path/to/agent.py"]) in chat_settings.json. Runs inside
#   app.py as a thread, or on its own: python studio/runner.py
import os, sys, json, time, threading, subprocess
LOKI = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.dirname(LOKI); sys.path.insert(0, LOKI)   # HERE: the repo, where results/ lives
from tools.local import load_queue, save_queue, QUEUE
OUT = os.path.join(HERE, "results", "studio", "out")

class Runner(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True); self.cfg = cfg; self.p = None; self.current = None; self.stop = False
    def agent_cmd(self): return list(self.cfg.get("studio_agent") or [sys.executable, os.path.join(LOKI, "studio", "stub_agent.py")])
    def update(self, jid, **fields):
        q = load_queue()
        for j in q:
            if j.get("id") == jid: j.update(fields)
        save_queue(q)
    def run(self):
        os.makedirs(OUT, exist_ok=True)
        while not self.stop:
            q = load_queue(); job = next((j for j in q if j.get("status", "").startswith("waiting")), None)
            if not job: time.sleep(1.5); continue
            self.current = job["id"]; self.update(job["id"], status="rendering · 0%", started=time.time())
            try: self.one(job)
            except Exception as e: self.update(job["id"], status=f"failed · {e}")
            self.current = None
    def one(self, job):
        job = dict(job); job["out"] = os.path.join(OUT, f"{job['id']}.{'mp4' if job.get('kind') == 'video' else 'png'}")
        self.p = subprocess.Popen(self.agent_cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=HERE)
        self.p.stdin.write("JOB " + json.dumps(job, ensure_ascii=False) + "\n"); self.p.stdin.flush()
        for line in self.p.stdout:
            head, _, rest = line.rstrip("\n").partition(" ")
            if head == "PROG": self.update(job["id"], status=f"rendering · {int(float(rest.split()[0]))}%" + (" · " + rest.split(" ", 1)[1] if " " in rest.strip() else ""))
            elif head == "DONE":
                path = rest.split(" ", 1)[0]; rel = os.path.relpath(path, HERE).replace(os.sep, "/")
                self.update(job["id"], status="done", out=rel, finished=time.time()); break
            elif head == "FAIL": self.update(job["id"], status=f"failed · {rest.strip()}"); break
        else: self.update(job["id"], status="failed · the agent stopped without a result")
        try: self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception: self.p.kill()
    def cancel(self, jid):   # drop a job: kill the agent if it is the one running
        if self.current == jid and self.p: self.p.kill()

if __name__ == "__main__":
    try: cfg = json.load(open(os.path.join(HERE, "chat_settings.json"), encoding="utf-8"))
    except Exception: cfg = {}
    r = Runner(cfg); r.start(); print("studio runner · watching", os.path.relpath(QUEUE, HERE), "· agent:", " ".join(r.agent_cmd()), flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: pass
