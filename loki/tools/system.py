# tools/system.py -- the machine itself: hardware, what is running, what is turning the fans over, a process and its
#   tree, ports, services, recent errors, the GPU, and search / grep over the disk. Windows through PowerShell and
#   nvidia-smi, Linux through /proc and ps. Everything is read-only except kill, which asks first.
import os, re, json, time, subprocess, shutil, platform, datetime
from . import tool, HERE

WIN = os.name == "nt"

def ps(script, timeout=60):   # one PowerShell call, JSON back
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, timeout=timeout)
    if r.returncode and not r.stdout.strip(): raise RuntimeError(r.stderr.strip()[:300] or "powershell failed")
    out = r.stdout.strip()
    try: j = json.loads(out) if out else []
    except Exception: return out
    return j if isinstance(j, list) else [j]

def sh(cmd, timeout=60):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout); return (r.stdout or "") + (r.stderr if r.returncode and not r.stdout else "")

def mb(b): return f"{b / 1048576:.0f} MB" if b < 10 * 1024 ** 3 else f"{b / 1024 ** 3:.1f} GB"

def nvidia(query, what="gpu"):
    if not shutil.which("nvidia-smi"): return []
    r = subprocess.run(["nvidia-smi", f"--query-{what}={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    return [[c.strip() for c in l.split(",")] for l in r.stdout.strip().splitlines() if l.strip()] if r.returncode == 0 else []

# ---- the box
@tool("sysinfo", "this machine: os, cpu, memory, disks, gpu, battery, uptime", {}, cap=3000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def sysinfo():
    out = [f"{platform.node()} · {platform.system()} {platform.release()} ({platform.version()}) · {platform.machine()}"]
    if WIN:
        cpu = ps("Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,LoadPercentage | ConvertTo-Json")
        for c in cpu: out.append(f"cpu: {c['Name'].strip()} · {c['NumberOfCores']} cores / {c['NumberOfLogicalProcessors']} threads · {c['MaxClockSpeed']} MHz · load {c.get('LoadPercentage')}%")
        osi = ps("Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory,LastBootUpTime | ConvertTo-Json")[0]
        tot, free = int(osi["TotalVisibleMemorySize"]) * 1024, int(osi["FreePhysicalMemory"]) * 1024
        out.append(f"memory: {mb(tot - free)} used of {mb(tot)}")
        boot = osi.get("LastBootUpTime", ""); m = re.search(r"/Date\((\d+)\)/", str(boot))
        if m: out.append(f"up since {datetime.datetime.fromtimestamp(int(m.group(1)) / 1000):%Y-%m-%d %H:%M}")
        for d in ps("Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select DeviceID,Size,FreeSpace,VolumeName | ConvertTo-Json"):
            if d.get("Size"): out.append(f"disk {d['DeviceID']} {d.get('VolumeName') or ''}: {mb(int(d['Size']) - int(d['FreeSpace']))} used of {mb(int(d['Size']))}")
        for b in ps("Get-CimInstance Win32_Battery | Select EstimatedChargeRemaining,BatteryStatus | ConvertTo-Json"):
            out.append(f"battery: {b.get('EstimatedChargeRemaining')}% · " + {1: "discharging", 2: "on power", 3: "full", 4: "low", 5: "critical", 6: "charging"}.get(b.get("BatteryStatus"), str(b.get("BatteryStatus"))))
        for g in ps("Get-CimInstance Win32_VideoController | Select Name,AdapterRAM,DriverVersion | ConvertTo-Json"):
            out.append(f"gpu: {g['Name']} · driver {g.get('DriverVersion')}")
    else:
        try: model = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        except Exception: model = platform.processor()
        out.append(f"cpu: {model} · {os.cpu_count()} threads · load {', '.join(f'{x:.2f}' for x in os.getloadavg())}")
        mi = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if ":" in l}
        out.append(f"memory: {mb(mi['MemTotal'] - mi.get('MemAvailable', 0))} used of {mb(mi['MemTotal'])}")
        up = float(open("/proc/uptime").read().split()[0]); out.append(f"up {int(up // 3600)}h {int(up % 3600 // 60)}m")
        st = shutil.disk_usage("/"); out.append(f"disk /: {mb(st.used)} used of {mb(st.total)}")
    for g in nvidia("name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw"):
        out.append(f"nvidia: {g[0]} · {g[1]} / {g[2]} MB · {g[3]}°C · {g[4]}% busy · {g[5]} W")
    return "\n".join(out)

# ---- what is running, and what is turning over
def sample_linux(secs):
    def snap():
        d = {}
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                st = open(f"/proc/{pid}/stat").read(); name = st[st.index("(") + 1:st.rindex(")")]; f = st[st.rindex(")") + 2:].split()
                rss = int(open(f"/proc/{pid}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                d[int(pid)] = (name, int(f[1]), int(f[11]) + int(f[12]), rss)   # name, ppid, ticks, rss
            except Exception: pass
        return d
    a = snap(); time.sleep(secs); b = snap(); hz = os.sysconf("SC_CLK_TCK"); n = os.cpu_count() or 1
    rows = []
    for pid, (name, ppid, ticks, rss) in b.items():
        cpu = (ticks - a[pid][2]) / hz / secs * 100 / n if pid in a else 0.0
        rows.append({"pid": pid, "name": name, "ppid": ppid, "cpu": cpu, "mem": rss})
    return rows

def sample_windows(secs):   # two Get-Process snapshots secs apart; parents from one Win32_Process query, looked up by hashtable (one WMI call, not one per process)
    script = f"""$pp=@{{}}; Get-CimInstance Win32_Process | % {{ $pp[[int]$_.ProcessId] = [int]$_.ParentProcessId }}
$a=@{{}}; Get-Process | % {{ $a[$_.Id] = $_.CPU }}
Start-Sleep -Seconds {secs}
$n=[Environment]::ProcessorCount
Get-Process | % {{ $o=$a[$_.Id]; $d= if($o -ne $null -and $_.CPU -ne $null){{($_.CPU-$o)/{secs}*100/$n}}else{{0}}; if($d -lt 0){{$d=0}}
 [pscustomobject]@{{pid=$_.Id;name=$_.ProcessName;ppid=$(if($pp.ContainsKey($_.Id)){{$pp[$_.Id]}}else{{0}});cpu=[math]::Round($d,1);mem=$_.WorkingSet64}} }} | ConvertTo-Json -Compress"""
    return ps(script, timeout=secs + 60)

def sample(secs): return sample_windows(secs) if WIN else sample_linux(secs)

@tool("processes", "what is running: top processes by cpu (sampled live) or memory, with pid and parent", {"sort": ("cpu or mem, default cpu", False), "n": ("how many, default 20", False), "filter": ("only names containing this", False), "seconds": ("how long to sample cpu, default 2", False)},
      cap=4000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def processes(sort="cpu", n=20, filter=None, seconds=2):
    rows = sample(max(1, min(int(seconds or 2), 10))); n = max(1, min(int(n or 20), 60))
    if filter: rows = [r for r in rows if str(filter).lower() in r["name"].lower()]
    rows.sort(key=lambda r: r["mem" if sort == "mem" else "cpu"], reverse=True)
    busy = sum(r["cpu"] for r in rows)
    head = f"{len(rows)} processes · {busy:.0f}% of all cores busy · top by {'memory' if sort == 'mem' else 'cpu'}"
    return head + "\n" + "\n".join(f"{r['pid']:>6}  {r['cpu']:5.1f}%  {mb(r['mem']):>8}  {r['name']}  (parent {r['ppid']})" for r in rows[:n])

@tool("load", "what is turning the fans over: a few seconds of cpu by process, gpu busy and temperature, cpu temperature and fans where the hardware reports them",
      {"seconds": ("how long to watch, default 3", False)}, cap=4000, summary=lambda a, t: t.split("\n", 1)[0][:70])
def load(seconds=3):
    secs = max(1, min(int(seconds or 3), 15)); rows = sample(secs); rows.sort(key=lambda r: r["cpu"], reverse=True)
    busy = sum(r["cpu"] for r in rows); out = [f"over {secs}s: cpu {busy:.0f}% busy across all cores"]
    for g in nvidia("name,utilization.gpu,temperature.gpu,fan.speed,power.draw,clocks.sm,memory.used"):
        out.append(f"gpu {g[0]}: {g[1]}% busy · {g[2]}°C · fan {g[3]}% · {g[4]} W · {g[5]} MHz · {g[6]} MB used")
    for p in nvidia("pid,process_name,used_memory", "compute-apps"): out.append(f"  on the gpu: pid {p[0]} {os.path.basename(p[1])} · {p[2]} MB")
    if WIN:
        try:
            for z in ps("Get-CimInstance -Namespace root/wmi MSAcpi_ThermalZoneTemperature | Select InstanceName,CurrentTemperature | ConvertTo-Json"):
                out.append(f"thermal zone {z.get('InstanceName', '').split(chr(92))[-1]}: {(int(z['CurrentTemperature']) - 2732) / 10:.0f}°C")
        except Exception: out.append("cpu temperature: not exposed by this firmware (ACPI thermal zone unavailable)")
        try:
            fans = ps("Get-CimInstance Win32_Fan | Select Name,DesiredSpeed,ActiveCooling | ConvertTo-Json")
            out.append("fans: " + (", ".join(f"{f.get('Name')} {f.get('DesiredSpeed') or '?'} rpm" for f in fans) if fans else "not exposed by this firmware (laptops seldom report fan rpm to Windows; the gpu fan above is from nvidia-smi)"))
        except Exception: pass
    else:
        for root, _, files in os.walk("/sys/class/hwmon"):
            for f in files:
                if re.match(r"temp\d+_input|fan\d+_input", f):
                    try: v = int(open(os.path.join(root, f)).read()); out.append(f"{os.path.basename(root)} {f.split('_')[0]}: {v / 1000:.0f}°C" if f.startswith("temp") else f"{f}: {v} rpm")
                    except Exception: pass
    out.append("busiest:"); out += [f"  {r['pid']:>6}  {r['cpu']:5.1f}%  {mb(r['mem']):>8}  {r['name']}" for r in rows[:12] if r["cpu"] > 0.4] or ["  nothing above half a percent"]
    return "\n".join(out)

@tool("proc", "one process in detail: path, command line, parent chain, children, threads, memory, cpu time, services it hosts, ports it holds", {"pid": ("the pid, or a name", True)}, cap=5000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def proc(pid):
    rows = sample(1); pid = str(pid)
    if not pid.isdigit():
        hits = [r for r in rows if pid.lower() in r["name"].lower()]
        if not hits: return f"nothing running called {pid}"
        if len(hits) > 1: return f"{len(hits)} processes called {pid}:\n" + "\n".join(f"{r['pid']:>6}  {r['cpu']:5.1f}%  {mb(r['mem']):>8}  {r['name']}" for r in hits[:30]) + "\nask again with a pid"
        pid = str(hits[0]["pid"])
    pid = int(pid); by = {r["pid"]: r for r in rows}
    if pid not in by: return f"no process {pid}"
    r = by[pid]; out = [f"{pid} {r['name']} · cpu now {r['cpu']:.1f}% · {mb(r['mem'])} working set"]
    if WIN:
        try:
            d = ps(f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | Select ExecutablePath,CommandLine,CreationDate,ThreadCount,KernelModeTime,UserModeTime | ConvertTo-Json")[0]
            out.append(f"path: {d.get('ExecutablePath') or '?'}"); out.append(f"command: {(d.get('CommandLine') or '')[:300]}")
            out.append(f"threads: {d.get('ThreadCount')} · cpu time {(int(d.get('KernelModeTime') or 0) + int(d.get('UserModeTime') or 0)) / 1e7:.0f}s")
        except Exception as e: out.append(f"details: {e}")
        try:
            svc = [l for l in sh(["tasklist", "/svc", "/fi", f"PID eq {pid}", "/fo", "csv", "/nh"]).splitlines() if l.strip()]
            if svc: out.append("services: " + svc[0].split('","')[-1].strip('"'))
        except Exception: pass
        ports = [l.split() for l in sh("netstat -ano").splitlines() if l.strip().endswith(f" {pid}") or l.rstrip().endswith(str(pid))]
        ports = [p for p in ports if len(p) >= 4 and p[-1] == str(pid)]
        if ports: out.append("ports: " + ", ".join(f"{p[1]} {p[3] if len(p) > 4 else ''}".strip() for p in ports[:12]))
    else:
        try:
            out.append(f"path: {os.readlink(f'/proc/{pid}/exe')}")
            out.append("command: " + open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()[:300])
            out.append(f"threads: {len(os.listdir(f'/proc/{pid}/task'))}")
        except Exception as e: out.append(f"details: {e}")
    chain = []; p = r
    while p and p["ppid"] in by and len(chain) < 8: p = by[p["ppid"]]; chain.append(f"{p['pid']} {p['name']}")
    if chain: out.append("parents: " + " ← ".join(chain))
    kids = [k for k in rows if k["ppid"] == pid]
    if kids: out.append(f"children ({len(kids)}):"); out += [f"  {k['pid']:>6}  {k['cpu']:5.1f}%  {mb(k['mem']):>8}  {k['name']}" for k in kids[:30]]
    return "\n".join(out)

@tool("ports", "listening ports and connections, with the process on each", {"filter": ("only lines containing this (a port, an address, a name)", False)}, cap=4000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def ports(filter=None):
    rows = sample(1); by = {r["pid"]: r["name"] for r in rows}; out = []
    if WIN:
        for l in sh("netstat -ano").splitlines():
            p = l.split()
            if len(p) >= 4 and p[0] in ("TCP", "UDP") and p[-1].isdigit(): out.append(f"{p[0]:<4} {p[1]:<24} {p[2] if p[0] == 'TCP' else '':<24} {p[3] if p[0] == 'TCP' else '':<12} {p[-1]:>6} {by.get(int(p[-1]), '')}")
    else:
        for l in sh("ss -tulpn 2>/dev/null || netstat -tulpn 2>/dev/null").splitlines()[1:]:
            if l.strip(): out.append(l)
    if filter: out = [l for l in out if str(filter).lower() in l.lower()]
    return f"{len(out)} entries\n" + "\n".join(out[:80]) if out else "nothing" + (f" matching {filter}" if filter else "")

@tool("services", "services and their state", {"filter": ("only names containing this", False), "state": ("running or stopped, default running", False)}, cap=4000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def services(filter=None, state="running"):
    if WIN:
        rows = ps("Get-Service | Select Name,DisplayName,Status,StartType | ConvertTo-Json"); want = {1: "stopped", 4: "running"}
        rows = [r for r in rows if want.get(r.get("Status"), str(r.get("Status"))) == (state or "running")]
        if filter: rows = [r for r in rows if str(filter).lower() in (r["Name"] + r["DisplayName"]).lower()]
        return f"{len(rows)} {state}\n" + "\n".join(f"{r['Name']:<28} {r['DisplayName'][:50]}" for r in rows[:80])
    out = sh(f"systemctl list-units --type=service --state={'running' if state != 'stopped' else 'exited,failed'} --no-pager --no-legend").splitlines()
    if filter: out = [l for l in out if str(filter).lower() in l.lower()]
    return f"{len(out)} {state}\n" + "\n".join(l.strip()[:120] for l in out[:80])

@tool("events", "recent errors and warnings from the system log", {"hours": ("how far back, default 24", False), "n": ("how many, default 30", False)}, cap=5000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def events(hours=24, n=30):
    hours = int(hours or 24); n = int(n or 30)
    if WIN:
        rows = ps(f"Get-WinEvent -FilterHashtable @{{LogName='System','Application'; Level=1,2,3; StartTime=(Get-Date).AddHours(-{hours})}} -MaxEvents {n} -ErrorAction SilentlyContinue | Select TimeCreated,LevelDisplayName,ProviderName,Id,Message | ConvertTo-Json")
        if not rows: return f"nothing in the last {hours}h"
        return "\n".join(f"{re.sub(r'/Date..(\d+)..', lambda m: datetime.datetime.fromtimestamp(int(m.group(1)) / 1000).strftime('%m-%d %H:%M'), str(r.get('TimeCreated')))} {r.get('LevelDisplayName')} {r.get('ProviderName')} #{r.get('Id')}: {' '.join(str(r.get('Message') or '').split())[:160]}" for r in rows)
    out = sh(f"journalctl -p warning --since '-{hours}h' --no-pager -n {n} 2>/dev/null || dmesg | tail -n {n}")
    return out.strip() or f"nothing in the last {hours}h"

@tool("gpu", "the nvidia gpu now: memory, busy, temperature, clocks, power, and which processes are on it", {}, cap=2000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def gpu():
    rows = nvidia("name,memory.used,memory.total,utilization.gpu,temperature.gpu,fan.speed,power.draw,power.limit,clocks.sm,clocks.mem,driver_version")
    if not rows: return "nvidia-smi is not on this machine"
    out = [f"{g[0]} · {g[1]} / {g[2]} MB · {g[3]}% busy · {g[4]}°C · fan {g[5]}% · {g[6]} / {g[7]} W · sm {g[8]} MHz · mem {g[9]} MHz · driver {g[10]}" for g in rows]
    out += [f"  pid {p[0]} {os.path.basename(p[1])} · {p[2]} MB" for p in nvidia("pid,process_name,used_memory", "compute-apps")] or ["  no compute processes on it"]
    return "\n".join(out)

@tool("kill", "end a process by pid", {"pid": ("the pid", True)}, cap=200, policy="ask")
def kill(pid):
    pid = int(pid)
    if WIN: return sh(["taskkill", "/PID", str(pid), "/T"]).strip() or f"sent {pid} a close"
    os.kill(pid, 15); return f"sent {pid} SIGTERM"

# ---- the disk
SKIP_DIRS = {"node_modules", ".git", "__pycache__", "$Recycle.Bin", "System Volume Information", "AppData", "Windows", ".cache", "site-packages", "ptx_trees"}
def roots(root):
    if root: return [os.path.expanduser(str(root))]
    if WIN: return [os.environ.get("USERPROFILE", "C:\\Users")]
    return [os.path.expanduser("~")]

@tool("find", "find files by name on this machine (a substring or a glob like *.pdf), newest first", {"pattern": ("name or glob", True), "root": ("folder to start in, default your home", False), "n": ("how many, default 40", False)}, cap=4000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def find(pattern, root=None, n=40):
    import fnmatch
    pat = str(pattern); glob = any(c in pat for c in "*?["); n = int(n or 40); hits = []; t0 = time.time(); scanned = 0
    for base in roots(root):
        for d, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
            for f in files:
                scanned += 1
                if (fnmatch.fnmatch(f.lower(), pat.lower()) if glob else pat.lower() in f.lower()):
                    p = os.path.join(d, f)
                    try: hits.append((os.path.getmtime(p), os.path.getsize(p), p))
                    except Exception: pass
            if time.time() - t0 > 25 or scanned > 400000: break
    hits.sort(reverse=True)
    if not hits: return f"nothing named {pattern} under {roots(root)[0]} ({scanned} files looked at)"
    return f"{len(hits)} found ({scanned} files looked at{', stopped early' if time.time() - t0 > 25 else ''})\n" + "\n".join(f"{datetime.datetime.fromtimestamp(m):%Y-%m-%d}  {mb(s):>8}  {p}" for m, s, p in hits[:n])

@tool("grep", "search inside files for a word or regex, with the matching lines", {"pattern": ("word or regex", True), "root": ("folder or file to search, default the repo", False), "ext": ("only these extensions, e.g. py,md,txt", False), "n": ("how many matches, default 40", False)}, cap=5000, summary=lambda a, t: t.split("\n", 1)[0][:60])
def grep(pattern, root=None, ext=None, n=40):
    rx = re.compile(str(pattern), re.I); base = os.path.expanduser(str(root)) if root else HERE; exts = {("." + e.strip().lstrip(".")).lower() for e in str(ext).split(",")} if ext else None
    n = int(n or 40); out = []; files = 0; t0 = time.time()
    walk = [(os.path.dirname(base), [], [os.path.basename(base)])] if os.path.isfile(base) else os.walk(base)
    for d, dirs, fs in walk:
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
        for f in fs:
            p = os.path.join(d, f)
            if exts and os.path.splitext(f)[1].lower() not in exts: continue
            try:
                if os.path.getsize(p) > 4 * 1024 * 1024: continue
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    files += 1
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            out.append(f"{os.path.relpath(p, base) if base != p else f}:{i}: {line.strip()[:160]}")
                            if len(out) >= n: break
            except Exception: continue
            if len(out) >= n or time.time() - t0 > 25: break
        if len(out) >= n or time.time() - t0 > 25: break
    return (f"{len(out)} matches in {files} files" + (" (stopped at the cap)" if len(out) >= n else "") + "\n" + "\n".join(out)) if out else f"no match for {pattern} in {files} files under {base}"
