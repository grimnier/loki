# tools/web.py -- the internet, with real hands: search, fetch any page as readable text (paged, searchable, with its
#   links), follow links, Wikipedia, raw HTTP for APIs, downloads, weather. Standard library only: urllib + html.parser.
#   Pages are cached for ten minutes so paging through a long one or pulling its links does not refetch it.
#   Search uses DuckDuckGo's HTML endpoint by default (no key); set BRAVE_API_KEY (env or chat_settings.json "keys")
#   and Brave's API is used instead, which is faster and gives cleaner snippets.
import os, re, json, time, html, gzip, io, urllib.request, urllib.parse, urllib.error, http.cookiejar
from html.parser import HTMLParser
from . import tool, HERE

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
TIMEOUT = 20; MAX_BYTES = 3 * 1024 * 1024; CACHE_TTL = 600
_jar = http.cookiejar.CookieJar(); _opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_jar))
_cache = {}   # url -> (time, Page)

def key(name):
    try: return os.environ.get(name) or (json.load(open(os.path.join(HERE, "chat_settings.json"), encoding="utf-8")).get("keys") or {}).get(name)
    except Exception: return os.environ.get(name)

def http(url, method="GET", headers=None, body=None, binary=False):
    """One request. Returns (status, headers, text-or-bytes). Follows redirects, accepts gzip, stops at MAX_BYTES."""
    if not re.match(r"^https?://", url): url = "https://" + url
    h = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/json,text/plain,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.8", "Accept-Encoding": "gzip"}
    h.update(headers or {})
    data = body.encode("utf-8") if isinstance(body, str) else body
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with _opener.open(req, timeout=TIMEOUT) as r:
            raw = r.read(MAX_BYTES + 1); status = r.status; rh = dict(r.headers)
    except urllib.error.HTTPError as e:
        raw = e.read(MAX_BYTES + 1); status = e.code; rh = dict(e.headers)
    if rh.get("Content-Encoding", "").lower() == "gzip":
        try: raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read(MAX_BYTES + 1)
        except Exception: pass
    cut = len(raw) > MAX_BYTES; raw = raw[:MAX_BYTES]
    if binary: return status, rh, raw
    ctype = rh.get("Content-Type", ""); m = re.search(r"charset=([\w-]+)", ctype)
    enc = m.group(1) if m else ("utf-8")
    try: text = raw.decode(enc, "replace")
    except Exception: text = raw.decode("utf-8", "replace")
    if cut: text += "\n[response cut at 3 MB]"
    return status, rh, text

# ---- HTML to readable text
BLOCK = {"p", "div", "section", "article", "main", "br", "li", "ul", "ol", "tr", "table", "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "figcaption", "hr", "header", "footer"}
SKIP = {"script", "style", "noscript", "svg", "canvas", "template", "iframe", "nav", "aside", "form", "button", "select", "option", "head"}
class Reader(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.out = []; self.links = []; self.title = ""; self.skip = 0; self.in_title = False; self.pre = 0
        self.main = []; self.in_main = 0; self.href = None; self.atext = []; self.cell = False
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title": self.in_title = True; return
        if tag in SKIP or (tag in ("header", "footer") and not self.in_main): self.skip += 1; return
        if self.skip: return
        if tag in ("main", "article") or a.get("role") == "main": self.in_main += 1
        if tag == "pre": self.pre += 1
        if tag in ("td", "th"): self.cell = True
        if tag == "a" and a.get("href"): self.href = a["href"]; self.atext = []
        if tag == "img" and a.get("alt"): self._emit(f"[image: {a['alt'].strip()}]")
        if tag in BLOCK: self._emit("\n")
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"): self._emit("\n" + "#" * int(tag[1]) + " ")
        if tag == "li": self._emit("- ")
        if tag == "hr": self._emit("\n---\n")
    def handle_endtag(self, tag):
        if tag == "title": self.in_title = False; return
        if tag in SKIP or (tag in ("header", "footer") and not self.in_main): self.skip = max(0, self.skip - 1); return
        if self.skip: return
        if tag in ("main", "article"): self.in_main = max(0, self.in_main - 1)
        if tag == "pre": self.pre = max(0, self.pre - 1)
        if tag in ("td", "th"): self.cell = False; self._emit(" | ")
        if tag == "a" and self.href is not None:
            t = " ".join("".join(self.atext).split())
            if t: self.links.append((t, self.href))
            self.href = None
        if tag in BLOCK and tag != "li": self._emit("\n")
    def handle_data(self, d):
        if self.in_title: self.title += d; return
        if self.skip: return
        if self.href is not None: self.atext.append(d)
        self._emit(d if self.pre else re.sub(r"\s+", " ", d))
    def _emit(self, s):
        self.out.append(s)
        if self.in_main: self.main.append(s)

class Page:
    def __init__(self, url, status, ctype, text, title, links):
        self.url, self.status, self.ctype, self.text, self.title, self.links = url, status, ctype, text, title, links

def tidy(s):
    s = re.sub(r"[ \t]+\n", "\n", s); s = re.sub(r"\n[ \t]+", "\n", s); s = re.sub(r" {2,}", " ", s); return re.sub(r"\n{3,}", "\n\n", s).strip()

def page(url, refresh=False):
    """Fetch and read a page (cached). Non-HTML comes back as its text (JSON pretty-printed)."""
    now = time.time(); c = _cache.get(url)
    if c and not refresh and now - c[0] < CACHE_TTL: return c[1]
    status, h, body = http(url); ctype = h.get("Content-Type", "").split(";")[0].strip()
    if "html" in ctype or (not ctype and body.lstrip()[:1] == "<"):
        r = Reader()
        try: r.feed(body); r.close()
        except Exception: pass
        text = tidy("".join(r.main)) if r.main and len("".join(r.main)) > 400 else tidy("".join(r.out))   # the article when the page marks one
        base = url; links = []
        seen = set()
        for t, href in r.links:
            u = urllib.parse.urljoin(base, href.strip())
            if u.startswith("http") and u not in seen and not u.startswith(url + "#"): seen.add(u); links.append((t[:80], u))
        p = Page(url, status, ctype, text, " ".join(r.title.split()), links)
    elif "json" in ctype:
        try: body = json.dumps(json.loads(body), indent=1, ensure_ascii=False)
        except Exception: pass
        p = Page(url, status, ctype, body, "", [])
    else: p = Page(url, status, ctype, body, "", [])
    _cache[url] = (now, p); return p

def window(text, start, n):   # a slice on a line boundary, with where it sits
    start = max(0, min(int(start or 0), len(text)))
    end = min(len(text), start + n)
    if end < len(text):
        nl = text.rfind("\n", start + n // 2, end)
        if nl > 0: end = nl
    foot = f"\n[characters {start}–{end} of {len(text)}" + (f" · more: start={end}]" if end < len(text) else " · end]")
    return text[start:end] + foot

# ---- tools
def ddg(q, n):
    status, _, body = http("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q}))
    out = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?(?:<a[^>]+class="result__snippet"[^>]*>(.*?)</a>)?', body, re.S):
        href = html.unescape(m.group(1)); qs = urllib.parse.urlparse(href).query
        real = urllib.parse.parse_qs(qs).get("uddg", [href])[0] if "uddg=" in qs else href
        title = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip(); snip = html.unescape(re.sub(r"<[^>]+>", "", m.group(3) or "")).strip()
        out.append((title, real, snip))
        if len(out) >= n: break
    if not out and status != 200: raise RuntimeError(f"duckduckgo answered {status}")
    return out

def brave(q, n):
    status, _, body = http("https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": q, "count": n}), headers={"X-Subscription-Token": key("BRAVE_API_KEY"), "Accept": "application/json", "Accept-Encoding": "identity"})
    if status != 200: raise RuntimeError(f"brave answered {status}")
    return [(r.get("title", ""), r.get("url", ""), re.sub(r"<[^>]+>", "", r.get("description", ""))) for r in json.loads(body).get("web", {}).get("results", [])[:n]]

def bing(q, n):
    status, _, body = http("https://www.bing.com/search?" + urllib.parse.urlencode({"q": q, "count": n, "setlang": "en"}))
    out = []
    for m in re.finditer(r'<li class="b_algo".*?<h2><a href="([^"]+)"[^>]*>(.*?)</a></h2>.*?(?:<p[^>]*>(.*?)</p>)?', body, re.S):
        title = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip(); snip = html.unescape(re.sub(r"<[^>]+>", "", m.group(3) or "")).strip()
        out.append((title, html.unescape(m.group(1)), snip))
        if len(out) >= n: break
    if not out and status != 200: raise RuntimeError(f"bing answered {status}")
    return out

@tool("search", "web search; returns titles, urls and snippets", {"q": ("the query", True), "n": ("how many, default 5", False)}, cap=3000, summary=lambda a, t: f"{t.count(chr(10) + chr(10)) + 1} results" if t and not t.startswith("no results") else "no results")
def search(q, n=5):
    n = max(1, min(int(n or 5), 10)); rows = []; errors = []
    for backend in ([brave] if key("BRAVE_API_KEY") else []) + [ddg, bing]:   # the first that answers wins
        try:
            rows = backend(q, n)
            if rows: break
        except Exception as e: errors.append(f"{backend.__name__}: {e}")
    if not rows: return "no results" + (" (" + "; ".join(errors) + ")" if errors else "")
    return "\n\n".join(f"{i + 1}. {t}\n{u}\n{s}".rstrip() for i, (t, u, s) in enumerate(rows))

@tool("fetch", "read a web page as text; long pages come in parts, ask for the next with start; find= shows the lines around a word",
      {"url": ("the page", True), "start": ("character offset to continue from, default 0", False), "find": ("a word or phrase to look for in the page", False), "refresh": ("true to refetch instead of using the cache", False)},
      cap=6000, summary=lambda a, t: (t.split("\n", 1)[0][:60] if t else ""))
def fetch(url, start=0, find=None, refresh=False):
    p = page(url, refresh=bool(refresh) and str(refresh).lower() != "false")
    head = (f"{p.title}\n" if p.title else "") + (f"[http {p.status}]\n" if p.status != 200 else "")
    if find:
        hits = [m.start() for m in re.finditer(re.escape(str(find)), p.text, re.I)]
        if not hits: return head + f'"{find}" is not on this page ({len(p.text)} characters)'
        parts = []; last = -1
        for h in hits[:8]:
            a = max(0, p.text.rfind("\n", 0, max(0, h - 300))); b = p.text.find("\n", h + 300); b = len(p.text) if b < 0 else b
            if a <= last: continue
            parts.append(f"[at {a}]\n" + p.text[a:b].strip()); last = b
        return head + f'{len(hits)} match{"es" if len(hits) != 1 else ""} for "{find}"\n\n' + "\n…\n".join(parts)
    return head + window(p.text, start, 5000)

@tool("links", "the links on a page, numbered, so one can be followed with fetch", {"url": ("the page", True), "filter": ("only links whose text or url contains this", False), "start": ("skip this many", False)}, cap=3000, summary=lambda a, t: f"{t.count(chr(10)) + 1} links")
def links(url, filter=None, start=0):
    p = page(url); rows = [(t, u) for t, u in p.links if not filter or str(filter).lower() in (t + u).lower()]
    start = int(start or 0); rows = rows[start:start + 40]
    if not rows: return "no links" + (f' matching "{filter}"' if filter else "")
    return "\n".join(f"{start + i + 1}. {t or '(no text)'} — {u}" for i, (t, u) in enumerate(rows)) + (f"\n[more: start={start + 40}]" if len(p.links) > start + 40 else "")

@tool("wiki", "a Wikipedia summary", {"q": ("the topic", True)}, cap=2500, summary=lambda a, t: t.split("\n", 1)[0][:60])
def wiki(q):
    s, _, body = http("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "opensearch", "search": q, "limit": 1, "format": "json"}), headers={"Accept-Encoding": "identity"})
    titles = json.loads(body)[1] if s == 200 else []
    if not titles: return f"nothing on Wikipedia for {q}"
    s, _, body = http("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(titles[0].replace(" ", "_")), headers={"Accept-Encoding": "identity"})
    j = json.loads(body); return f"{j.get('title', titles[0])}\n{j.get('extract', '')}\n{j.get('content_urls', {}).get('desktop', {}).get('page', '')}"

@tool("http", "a raw HTTP request for APIs: method, url, headers, body; returns status and body", {"url": ("the url", True), "method": ("GET, POST, …; default GET", False), "headers": ("a json object of headers", False), "body": ("the request body", False)}, cap=6000, policy="ask", summary=lambda a, t: t.split("\n", 1)[0])
def http_tool(url, method="GET", headers=None, body=None):
    if isinstance(headers, str):
        try: headers = json.loads(headers)
        except Exception: headers = {}
    if isinstance(body, (dict, list)): body = json.dumps(body); headers = dict(headers or {}); headers.setdefault("Content-Type", "application/json")
    s, h, text = http(url, method=(method or "GET").upper(), headers=headers, body=body)
    if "json" in h.get("Content-Type", ""):
        try: text = json.dumps(json.loads(text), indent=1, ensure_ascii=False)
        except Exception: pass
    return f"http {s} · {h.get('Content-Type', '').split(';')[0]}\n{text}"

@tool("download", "save a file from a url into results/downloads", {"url": ("the file", True), "name": ("file name, default from the url", False)}, cap=500, policy="ask", summary=lambda a, t: t.split("\n", 1)[0])
def download(url, name=None):
    s, h, raw = http(url, binary=True)
    if s != 200: return f"http {s}"
    name = name or os.path.basename(urllib.parse.urlparse(url).path) or "download"
    name = re.sub(r"[^\w.\-]+", "_", name)[:120]; d = os.path.join(HERE, "results", "downloads"); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name); open(path, "wb").write(raw)
    return f"saved {len(raw)} bytes to results/downloads/{name} ({h.get('Content-Type', '').split(';')[0]})"

@tool("weather", "current weather and a short forecast for a place", {"place": ("city or place name", True)}, cap=1200, summary=lambda a, t: t.split("\n", 1)[0][:60])
def weather(place):
    s, _, body = http("https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({"name": place, "count": 1}), headers={"Accept-Encoding": "identity"})
    r = (json.loads(body).get("results") or [None])[0] if s == 200 else None
    if not r: return f"no place called {place}"
    s, _, body = http("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({"latitude": r["latitude"], "longitude": r["longitude"], "current": "temperature_2m,apparent_temperature,wind_speed_10m,precipitation,weather_code",
                      "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max", "forecast_days": 3, "timezone": "auto"}), headers={"Accept-Encoding": "identity"})
    j = json.loads(body); c = j.get("current", {}); d = j.get("daily", {})
    name = ", ".join(x for x in (r.get("name"), r.get("admin1"), r.get("country")) if x)
    out = [f"{name} · now {c.get('temperature_2m')}°C (feels {c.get('apparent_temperature')}°C) · wind {c.get('wind_speed_10m')} km/h · precipitation {c.get('precipitation')} mm"]
    for i, day in enumerate(d.get("time", [])): out.append(f"{day}: {d['temperature_2m_min'][i]}–{d['temperature_2m_max'][i]}°C · rain {d['precipitation_probability_max'][i]}%")
    return "\n".join(out)
