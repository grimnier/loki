# studio/stub_agent.py -- stands in for the 4K picture / video model so the queue, the runner, the progress and the
#   finished strip all work today. Reads one JOB line, reports progress for a few seconds, writes a placeholder:
#   a PNG (standard library, zlib) whose colours come from the prompt, at the requested aspect, 640 px wide;
#   a "video" is the same picture saved beside a .txt that says what the clip would be. Replace with the real agent.
import sys, os, json, time, zlib, struct, hashlib

def png(path, w, h, pix):   # pix(x, y) -> (r, g, b)
    raw = b"".join(b"\x00" + bytes(c for x in range(w) for c in pix(x, y)) for y in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))

def main():
    line = sys.stdin.readline()
    if not line.startswith("JOB "): print("FAIL no job", flush=True); return
    job = json.loads(line[4:]); out = job.get("out") or os.path.join("results", "studio", "out", f"{job.get('id', 'job')}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    seed = hashlib.sha256((job.get("prompt", "") + str(job.get("seed", ""))).encode()).digest()
    a, b = [(seed[i], seed[i + 1], seed[i + 2]) for i in (0, 3)]
    size = str(job.get("size", "3840 × 2160")).replace("×", "x").replace(" ", "").split("x")
    try: ratio = int(size[0]) / int(size[1])
    except Exception: ratio = 16 / 9
    w = 640; h = max(64, int(w / ratio))
    steps = max(4, min(int(job.get("steps") or 30), 60))
    for i in range(steps):                                            # "denoising"
        time.sleep(max(0.05, 4.0 / steps)); print(f"PROG {int(100 * (i + 1) / steps)} step {i + 1}/{steps}", flush=True)
    def pix(x, y):
        t = (x / w * 0.6 + y / h * 0.4); return tuple(int(a[k] * (1 - t) + b[k] * t) for k in range(3))
    path = out if out.endswith(".png") else out.rsplit(".", 1)[0] + ".png"
    png(path, w, h, pix)
    if job.get("kind") == "video": open(out.rsplit(".", 1)[0] + ".txt", "w", encoding="utf-8").write(f"placeholder for a {job.get('dur', '4s')} clip: {job.get('prompt', '')}\n")
    print(f"DONE {path} placeholder", flush=True)

if __name__ == "__main__": main()
