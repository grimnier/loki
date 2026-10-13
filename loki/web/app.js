// loki window: renders the state the server streams (log, turn, cfg) and sends what you type. No framework.
const $ = (s) => document.querySelector(s), $$ = (s) => Array.from(document.querySelectorAll(s));
const scroll = $("#scroll"), log = $("#log"), hood = $("#hood"), head = $("#hood-head"), thought = $("#thought");
const input = $("#input"), stop = $("#stop"), hint = $("#hint"), toBottom = $("#to-bottom");
let state = null, open = false, followThought = true, closing = null, tick = null, warm = null, first = true, prevTurn = false, hist = [], histI = null, palI = 0;
const HINT = hint.textContent;

const post = (path, body) => fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
const fmt = (s) => { s = Math.round(s); return s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`; };
const nearBottom = (el) => el.scrollHeight - el.scrollTop - el.clientHeight < 40;
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const store = { get(k) { try { return JSON.parse(localStorage.getItem("loki." + k)); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem("loki." + k, JSON.stringify(v)); } catch (e) {} } };

// ---- markdown, the little the model uses: fenced code, inline code, bold, italic, lists, paragraphs
function md(src) {
  const out = []; const lines = src.split("\n"); let i = 0, para = [], list = null;
  const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>")
    .replace(/(^|[\s(>])(https?:\/\/[^\s<)]+[^\s<).,;:!?])/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');   // bare urls open in a tab
  const flushP = () => { if (para.length) { out.push("<p>" + para.map(inline).join("<br>") + "</p>"); para = []; } };
  const flushL = () => { if (list) { out.push(`<${list.tag}>` + list.items.map((t) => "<li>" + inline(t) + "</li>").join("") + `</${list.tag}>`); list = null; } };
  while (i < lines.length) {
    const l = lines[i];
    if (/^```/.test(l)) {
      flushP(); flushL(); const code = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]);
      out.push("<pre><code>" + esc(code.join("\n")) + "</code></pre>"); i++; continue;
    }
    const m = /^\s*(?:([-*•])|(\d+)[.)])\s+(.*)$/.exec(l);
    if (m) { flushP(); const tag = m[2] ? "ol" : "ul"; if (!list || list.tag !== tag) { flushL(); list = { tag, items: [] }; } list.items.push(m[3]); i++; continue; }
    if (!l.trim()) { flushP(); flushL(); i++; continue; }
    if (list) { list.items[list.items.length - 1] += " " + l.trim(); i++; continue; }   // a wrapped list item
    para.push(l); i++;
  }
  flushP(); flushL(); return out.join("");
}

// text that is still arriving: append the new piece so it eases in, instead of repainting the whole thing
function arrive(el, text) {
  const old = el.dataset.raw || "";
  if (text.startsWith(old) && old) { const s = document.createElement("span"); s.className = "arrive"; s.textContent = text.slice(old.length); el.appendChild(s); }
  else el.textContent = text;
  el.dataset.raw = text;
}

// ---- conversation
const foldable = (e) => e.kind === "tool" && e.result !== undefined;   // "→ tool …" lines fold open to what came back. "thought for Ns" does not: the TV is off (the owner's rule)
function noteEl(e) {
  const n = document.createElement("div"); n.className = "msg tool thought-note"; n.dataset.kind = e.kind; n.dataset.thought = "1"; n.dataset.raw = e.text; n.textContent = e.text;
  const t = document.createElement("div"); t.className = "thought-text";
  const body = e.result || "(nothing came back)";
  if (e.kind === "tool" && /^→ (search|links|browse_links) /.test(e.text) && /^\d+\. /m.test(body)) t.appendChild(cards(body));   // results with urls: cards, the page one click away
  else t.textContent = body;
  n.appendChild(t); return n;
}
function cards(body) {
  const wrap = document.createElement("div"); wrap.className = "cards";
  for (const block of body.split(/\n(?=\d+\. )/)) {
    const m = /^(\d+)\. ([\s\S]*?)(?:\n|\s—\s)(https?:\/\/\S+)(?:\n([\s\S]*))?$/.exec(block.trim());
    if (!m) { const p = document.createElement("div"); p.className = "card-note"; p.textContent = block.trim(); if (block.trim()) wrap.appendChild(p); continue; }
    const c = document.createElement("a"); c.className = "card"; c.href = m[3]; c.target = "_blank"; c.rel = "noopener";
    c.innerHTML = `<b></b><span class="url"></span><span class="snip"></span>`; c.children[0].textContent = m[2].trim(); c.children[1].textContent = m[3].replace(/^https?:\/\//, "").slice(0, 80); c.children[2].textContent = (m[4] || "").trim();
    wrap.appendChild(c);
  }
  return wrap;
}
function renderLog(entries) {
  const kids = log.children;
  while (kids.length > entries.length) log.lastElementChild.remove();
  entries.forEach((e, i) => {
    let el = kids[i];
    if (!el || el.dataset.kind !== e.kind || !!el.dataset.thought !== foldable(e) || (foldable(e) && el.dataset.raw !== e.text)) {
      const n = foldable(e) ? noteEl(e) : document.createElement("div");
      if (!foldable(e)) { n.className = "msg " + e.kind; n.dataset.kind = e.kind; }
      if (!first) n.classList.add("new");                              // a new line eases in; on a reload everything is simply there
      if (el) el.replaceWith(n); else log.appendChild(n); el = n;
      if (foldable(e)) return;
    }
    const streaming = e.kind === "answer" && i === entries.length - 1 && state && state.turn;
    if (el.dataset.raw === e.text && !(el.classList.contains("streaming") && !streaming)) return;
    if (streaming) { el.classList.add("streaming"); arrive(el, e.text); return; }
    el.classList.remove("streaming"); el.dataset.raw = e.text;
    if (e.kind === "answer") el.innerHTML = md(e.text) + `<button class="copy" type="button" title="copy">copy</button>`;
    else if (e.kind === "user") {                                      // your message: edit it, or ask it again (the old tail is kept as a branch)
      el.textContent = e.text; const acts = document.createElement("span"); acts.className = "acts";
      acts.innerHTML = `<button type="button" data-act="edit">edit</button> <button type="button" data-act="again">again</button>`; el.appendChild(acts);
    }
    else el.textContent = e.text;
  });
}
async function editMessage(n, resend) {                               // the Nth-last message back into the box; resend asks it again as it is
  if (state && state.turn) return;
  const r = await (await post("/edit", { n })).json(); if (!r.text) return;
  if (resend) { post("/send", { text: r.text }); return; }
  input.value = r.text; grow(input); input.focus(); input.setSelectionRange(r.text.length, r.text.length);
}
document.addEventListener("click", (e) => {
  const act = e.target.closest("[data-act]");
  if (act) { const users = $$("#log .msg.user"); editMessage(users.length - users.indexOf(act.closest(".msg.user")), act.dataset.act === "again"); return; }   // N = how many user messages from this one to the end
  const c = e.target.closest(".copy");
  if (c) { navigator.clipboard.writeText(c.parentElement.dataset.raw).then(() => { c.textContent = "copied"; setTimeout(() => (c.textContent = "copy"), 1200); }); return; }
  const n = e.target.closest(".thought-note");
  if (n && !e.target.closest(".thought-text")) toggleNote(n);
});
function toggleNote(n) {
  const t = n.querySelector(".thought-text"); const on = !t.classList.contains("open");
  $$(".thought-text.open").forEach((o) => o !== t && o.classList.remove("open"));
  t.classList.toggle("open", on);
  if (n.closest("#log")) { store.set("open", on ? Array.from(log.children).indexOf(n) : -1); savePlace(); }
  if (on) requestAnimationFrame(() => { if (!nearBottom(scroll)) t.scrollIntoView({ block: "nearest", behavior: "smooth" }); });
}

// ---- the hood
function tickHead() {
  const t = state && state.turn; if (!t) return;
  if (t.tool) head.textContent = `${t.tool.text} · ${fmt(Date.now() / 1000 - t.tool.t0)}`;
  else if (t.in_think) head.textContent = `thinking · ${fmt(Date.now() / 1000 - t.t0)}` + (t.stopped ? " · stopping" : "");
}
const ask = $("#ask");
function renderHood(t) {
  ask.hidden = !(t && t.pending);
  if (t && t.pending) $("#ask-text").textContent = `run ${t.pending.text} ?`;
  if (t && (t.in_think || t.tool)) {
    if (closing) { clearTimeout(closing); closing = null; hood.classList.remove("closing"); }
    hood.hidden = false; tickHead();
    const text = t.tool ? "" : t.think;
    if (thought.dataset.raw !== text) {
      arrive(thought, text);
      if (!open || followThought) thought.scrollTop = thought.scrollHeight;
    }
    if (!tick) tick = setInterval(() => { tickHead(); setTitle(); }, 1000);
  } else if (!hood.hidden && !closing) {                           // the hood comes down over the thought, then it is gone
    hood.classList.add("closing"); open = false; thought.classList.remove("open");
    closing = setTimeout(() => { hood.hidden = true; hood.classList.remove("closing"); thought.textContent = ""; thought.dataset.raw = ""; closing = null; }, 320);
    if (tick) { clearInterval(tick); tick = null; }
  }
}
function toggleHood() {
  if (state && state.turn && state.turn.in_think) {
    open = !open; followThought = true; thought.classList.toggle("open", open);
    requestAnimationFrame(() => { thought.scrollTop = thought.scrollHeight; });
  }                                                                   // idle: nothing. A finished thought stays off — "thought for Ns" is all that remains
}
head.addEventListener("click", toggleHood);
thought.addEventListener("scroll", () => { if (open) followThought = nearBottom(thought); });

// ---- state in
function setTitle() {
  const t = state && state.turn;
  document.title = "loki" + (t ? " · " + (t.pending ? "a tool asks" : t.tool ? t.tool.text : t.in_think ? "thinking " + fmt(Date.now() / 1000 - t.t0) : "answering") : "");
}
function warmup() {
  if (!state || state.ready || state.dead) { if (warm) { clearInterval(warm); warm = null; } hint.textContent = HINT; return; }
  hint.textContent = `loading the model · ${fmt(Date.now() / 1000 - state.t_start)}`;
  if (!warm) warm = setInterval(warmup, 1000);
}
function apply(msg) {                                                 // a delta: every top-level field, and the log from the first changed entry on
  const stick = nearBottom(scroll);
  const s = msg.full || !state ? { ...msg } : { ...state, ...msg };
  s.log = (msg.full || !state ? [] : state.log.slice(0, msg.log_from)).concat(msg.log); delete s.log_from; delete s.full;
  state = s;
  const had = prevTurn; prevTurn = !!s.turn;
  renderLog(s.log); renderHood(s.turn); setTitle(); warmup();
  stop.hidden = !s.turn; input.placeholder = s.dead ? "engine stopped" : ""; $("#dead").hidden = !s.dead; hint.hidden = !!s.dead;
  hint.classList.toggle("gone", s.log.length > 0 && s.ready && !hint.classList.contains("warn"));   // the hint has done its job once a conversation exists
  const ql = $("#queue-lines"); ql.innerHTML = ""; for (const q of s.queue || []) { const d = document.createElement("div"); d.textContent = "queued · " + q; ql.appendChild(d); }
  hist = s.log.filter((e) => e.kind === "user").map((e) => e.text); histI = null;   // what you sent, for Up to recall
  if (had && !s.turn && !document.hasFocus()) notifyDone(s);        // an answer landed while you were elsewhere
  document.documentElement.style.setProperty("--bars", s.cfg.bars);
  $("#ctx-text").textContent = s.ready ? `${s.pos} / ${s.ctx}` : ""; $("#ctx b").style.width = s.ready ? `${Math.min(100, 100 * s.pos / s.ctx)}%` : "0";
  fillKnobs(s.cfg);
  if (first) { first = false; resume(); return; }
  if (stick) scroll.scrollTop = scroll.scrollHeight;
}
function resume() {                                                   // back after a reload: same conversation, same place
  const r = store.get("place");
  if (r && r.len === state.log.length) {
    if (r.open >= 0 && log.children[r.open]) log.children[r.open].querySelector(".thought-text").classList.add("open");
    scroll.scrollTop = r.top;
  } else scroll.scrollTop = scroll.scrollHeight;
  const v = store.get("view"); showView(document.getElementById(v) && v !== "chats" ? v : "chat");
}
let placeTimer = null;
function savePlace() { if (state) store.set("place", { len: state.log.length, top: scroll.scrollTop, open: store.get("open") ?? -1 }); }
scroll.addEventListener("scroll", () => { toBottom.hidden = nearBottom(scroll); clearTimeout(placeTimer); placeTimer = setTimeout(savePlace, 200); });
function connect() {
  const es = new EventSource("/events");
  es.onopen = () => { hint.textContent = HINT; hint.classList.remove("warn"); };
  es.onmessage = (ev) => apply(JSON.parse(ev.data));
  es.onerror = () => { es.close(); hint.textContent = "reconnecting"; hint.classList.add("warn"); setTimeout(connect, 1500); };
}
connect();
toBottom.addEventListener("click", () => scroll.scrollTo({ top: scroll.scrollHeight, behavior: "smooth" }));

// ---- composer
function grow(ta) { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + "px"; }
input.addEventListener("input", () => grow(input));
// the palette: "/" and the commands that match what is typed; Up/Down pick, Tab completes, Enter takes the selection
const palette = $("#palette");
function paletteRows() { const v = input.value; return v.startsWith("/") && !/[\s]/.test(v) && state ? (state.cmds || []).filter((c) => c.name.startsWith(v)) : []; }
function renderPalette() {
  const rows = paletteRows(); palette.hidden = !rows.length; if (!rows.length) return;
  palI = Math.min(palI, rows.length - 1); palette.innerHTML = "";
  rows.forEach((c, i) => { const d = document.createElement("div"); d.className = i === palI ? "on" : ""; d.innerHTML = "<span></span><span></span>"; d.children[0].textContent = (c.name + " " + c.args).trim(); d.children[1].textContent = c.desc;
    d.addEventListener("mousedown", (e) => { e.preventDefault(); input.value = c.name + (c.args ? " " : ""); grow(input); renderPalette(); input.focus(); }); palette.appendChild(d); });
}
input.addEventListener("input", () => { palI = 0; renderPalette(); histI = null; });
function recall(d) {                                                  // Up on an empty box: what you sent, older each press; Down comes back
  if (!hist.length) return; if (histI === null) { if (d > 0) return; histI = hist.length - 1; }
  else { const i = histI + d; if (i >= hist.length) { histI = null; input.value = ""; grow(input); return; } histI = Math.max(0, i); }
  input.value = hist[histI]; grow(input); input.setSelectionRange(input.value.length, input.value.length);
}
input.addEventListener("keydown", (e) => {
  const rows = paletteRows();
  if (rows.length && (e.key === "ArrowUp" || e.key === "ArrowDown")) { e.preventDefault(); palI = (palI + (e.key === "ArrowDown" ? 1 : rows.length - 1)) % rows.length; renderPalette(); return; }
  if (rows.length && e.key === "Tab") { e.preventDefault(); input.value = rows[palI].name + (rows[palI].args ? " " : ""); grow(input); renderPalette(); return; }
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (rows.length && input.value !== rows[palI].name) input.value = rows[palI].name; send(); }
  else if (e.key === "Tab") { e.preventDefault(); toggleHood(); }
  else if (e.key === "Escape") { if (state && state.turn) post("/stop"); else if (input.value) { input.value = ""; grow(input); renderPalette(); } else if (state && state.queue && state.queue.length) post("/queue"); }
  else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && (histI !== null || !input.value) && !input.value.includes("\n")) { e.preventDefault(); recall(e.key === "ArrowUp" ? -1 : 1); }
});
function notifyDone(s) {                                              // a system notification when an answer lands and the window is not in front
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  const last = s.log.filter((e) => e.kind === "answer").pop(); const body = last ? last.text.replace(/\s+/g, " ").slice(0, 120) : "done";
  try { const n = new Notification("loki", { body, silent: true }); setTimeout(() => n.close(), 8000); n.onclick = () => window.focus(); } catch (e) {}
}
function send() {
  const text = input.value.trim(); if (!text) return;
  if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();   // asked once, on the first send
  input.value = ""; grow(input); renderPalette(); post("/send", { text }); store.set("open", -1); scroll.scrollTop = scroll.scrollHeight;
  if (document.body.classList.contains("compact")) setTimeout(closeVisor, 260);   // the visor swings back up once the message is away
}
stop.addEventListener("click", () => post("/stop"));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { if (state && state.turn && e.target !== input) post("/stop"); if (!$("#knobs").hidden) $("#knobs").hidden = true; }   // the box handles its own Esc
  if (e.target === document.body && e.key.length === 1 && !e.ctrlKey && !e.metaKey && $("#chat").classList.contains("on")) { openVisor(); input.focus(); }   // typing anywhere lands in the box
  if (e.key === "." && e.ctrlKey) { e.preventDefault(); setCompact(document.body.classList.contains("compact") ? "off" : "on"); }
  if (e.key === "," && e.ctrlKey) { e.preventDefault(); knobs.hidden = !knobs.hidden; if (!knobs.hidden) fillKnobs(state && state.cfg, true); }   // knobs from the keyboard, so compact mode has them too
  if ((e.key === "b" || e.key === "B") && e.ctrlKey) { e.preventDefault(); toggleDrawer(); }
  if (e.key === "Escape" && document.body.classList.contains("compact")) { closeDrawer(); if (!(state && state.turn)) closeVisor(); }
});

// ---- rail
function showView(id) {
  $$(".tab[data-view]").forEach((t) => t.classList.toggle("on", t.dataset.view === id));
  $$(".view").forEach((v) => v.classList.toggle("on", v.id === id));
  store.set("view", id);
  if (id === "chat") input.focus();
  if (id === "chats") loadChats();
}
$$(".tab[data-view]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
$("#new").addEventListener("click", () => { post("/new"); store.set("open", -1); showView("chat"); });

// ---- knobs: the settings, same as the / commands
const knobs = $("#knobs");
$("#knobs-btn").addEventListener("click", () => { knobs.hidden = !knobs.hidden; if (!knobs.hidden) fillKnobs(state && state.cfg, true); });
document.addEventListener("mousedown", (e) => { if (!knobs.hidden && !knobs.contains(e.target) && !$("#knobs-btn").contains(e.target)) knobs.hidden = true; });
function fillKnobs(cfg, force) {
  if (!cfg) return;
  for (const k of ["think", "answer", "bars", "ctx", "hops", "tool_cap"]) { const el = $("#k-" + k); if (force || document.activeElement !== el) el.value = cfg[k] ?? ""; }
  $("#k-stats").classList.toggle("on", !!cfg.stats);
  $$("#k-tools button").forEach((b) => b.classList.toggle("on", b.dataset.v === String(cfg.tools ?? "auto")));
  const ts = state && state.tools; if (!ts) return;
  $("#k-tools-note").textContent = ts.enabled ? `${ts.list.filter((t) => t.policy !== "off").length} tools offered` : (cfg.tools === "auto" ? "auto: off until the context is 2048+" : "off");
  const belt = $("#belt"); const sig = JSON.stringify([ts.list, ts.rules]);
  if (belt.dataset.sig === sig) return; belt.dataset.sig = sig; belt.innerHTML = "";
  for (const t of ts.list) {
    const b = document.createElement("button"); b.type = "button"; b.textContent = t.name + (t.policy === "ask" ? " ?" : ""); b.dataset.p = t.policy; b.title = `${t.desc} · ${t.policy}`;
    b.addEventListener("click", () => post("/send", { text: `/tool ${t.name} ${{ allow: "ask", ask: "off", off: "allow" }[t.policy]}` })); belt.appendChild(b);
  }
  const rules = $("#rules"); rules.innerHTML = "";                  // remembered answers: click one to forget it
  (ts.rules || []).forEach((r, i) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = `${r.tool} ${r.match} → ${r.policy} ×`; b.title = "forget this rule";
    b.addEventListener("click", () => post("/send", { text: `/rules ${i + 1}` })); rules.appendChild(b);
  });
}
for (const k of ["think", "answer", "bars", "ctx", "hops", "tool_cap"]) {
  const el = $("#k-" + k);
  const commit = () => { const v = parseInt(el.value, 10); if (!isNaN(v) && state && v !== state.cfg[k]) post("/send", { text: `/${k === "tool_cap" ? "cap" : k} ${v}` }); };
  el.addEventListener("change", commit); el.addEventListener("keydown", (e) => { if (e.key === "Enter") { commit(); el.blur(); } });
}
$("#k-stats").addEventListener("click", () => post("/send", { text: "/stats" }));
$$("#k-tools button").forEach((b) => b.addEventListener("click", () => post("/send", { text: `/tools ${{ true: "on", false: "off", auto: "auto" }[b.dataset.v]}` })));
$("#ask-yes").addEventListener("click", () => post("/tool", { answer: "allow" }));
$("#ask-always").addEventListener("click", () => post("/tool", { answer: "always" }));
$("#ask-no").addEventListener("click", () => post("/tool", { answer: "deny" }));

// ---- conversations: every one is a file on the server; click to continue it, × to forget it
async function loadChats(list = $("#chats-list"), q = "") {           // fills the chats view (all, or the ones matching q), or the drawer
  let convs = [];
  try { convs = await (await fetch(q ? "/find?q=" + encodeURIComponent(q) : "/conversations")).json(); } catch (e) {}
  list.innerHTML = convs.length ? "" : `<div class="dim small">${q ? "nothing has " + esc(q) : "nothing yet"}</div>`;
  let day = "";
  for (const c of convs) {
    const d = new Date((c.updated || c.created || 0) * 1000); const dd = d.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
    if (dd !== day && !q) { day = dd; const h = document.createElement("div"); h.className = "chat-day"; h.textContent = dd; list.appendChild(h); }
    const row = document.createElement("div"); row.className = "chat-row dim" + (state && state.conv && state.conv.id === c.id ? " now" : "");
    row.innerHTML = `<span></span><span class="n"></span><button class="x" type="button" title="forget">×</button>` + (c.snippet ? `<span class="snip"></span>` : "");
    row.firstElementChild.textContent = c.title || "untitled";
    row.children[1].textContent = `${d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })} · ${c.turns || 0} turn${c.turns === 1 ? "" : "s"}` + (c.branches ? ` · ${c.branches} branch${c.branches === 1 ? "" : "es"}` : "");
    if (c.snippet) row.lastElementChild.textContent = "…" + c.snippet + "…";
    row.addEventListener("click", (e) => { if (e.target.closest(".x")) return; post("/open", { id: c.id }); showView("chat"); closeDrawer(); });
    row.querySelector(".x").addEventListener("click", async () => { await post("/delete", { id: c.id }); loadChats(list, q); });
    list.appendChild(row);
  }
}
let findTimer = null;
$("#chats-find").addEventListener("input", () => { clearTimeout(findTimer); findTimer = setTimeout(() => loadChats($("#chats-list"), $("#chats-find").value.trim()), 250); });
$("#restart").addEventListener("click", () => post("/send", { text: "/restart" }));

// ---- compact: under 880 × 560 (or when asked) the conversation has the whole window. The composer is a visor hinged on the
//      top edge: it swings down when you type, hover the top edge, press / or click the pill, and back up when you send or
//      press Esc. Conversations are a drawer hinged on the left edge: hover the edge, click the pill, or Ctrl-B.
const small = matchMedia("(max-width: 880px), (max-height: 560px)"); const composerWrap = $(".composer-wrap"), drawer = $("#drawer"), shade = $("#shade");
function setCompact(v) { store.set("compact", v); applyCompact(); }
function applyCompact() {
  const want = store.get("compact") || "auto"; const on = want === "on" || (want === "auto" && small.matches);
  const was = document.body.classList.contains("compact"); document.body.classList.toggle("compact", on);
  $$("#k-compact button").forEach((b) => b.classList.toggle("on", b.dataset.v === want));
  if (on && !was) { closeVisor(); closeDrawer(); } else if (!on && was) { composerWrap.classList.remove("open"); drawer.classList.remove("open"); shade.hidden = true; }
}
function openVisor() { if (!document.body.classList.contains("compact")) return; composerWrap.classList.add("open"); document.body.classList.add("visor"); }
function closeVisor() { if (document.activeElement === input) input.blur(); composerWrap.classList.remove("open"); document.body.classList.remove("visor"); }
function toggleDrawer() { drawer.classList.contains("open") ? closeDrawer() : openDrawer(); }
function openDrawer() { if (!document.body.classList.contains("compact")) { showView("chats"); return; } loadChats($("#drawer-list")); drawer.classList.add("open"); shade.hidden = false; }
function closeDrawer() { drawer.classList.remove("open"); shade.hidden = true; }
small.addEventListener("change", applyCompact); applyCompact();
$$("#k-compact button").forEach((b) => b.addEventListener("click", () => setCompact(b.dataset.v)));
$("#visor-zone").addEventListener("mouseenter", openVisor); $("#visor-handle").addEventListener("click", () => { openVisor(); input.focus(); });
$("#drawer-zone").addEventListener("mouseenter", openDrawer); $("#drawer-handle").addEventListener("click", toggleDrawer);
$("#drawer-new").addEventListener("click", () => { post("/new"); closeDrawer(); });
shade.addEventListener("click", () => { closeDrawer(); closeVisor(); });
scroll.addEventListener("mousedown", () => { if (document.body.classList.contains("compact") && !input.value) closeVisor(); });   // a click on the conversation folds the box away
input.addEventListener("keydown", (e) => { if (e.key.length === 1 && !e.ctrlKey && !e.metaKey) openVisor(); });   // the box has focus but is folded away: a keystroke brings it down

// ---- studio (rough: the controls and a queue, nothing behind them yet)
const sinput = $("#studio-input"); sinput.addEventListener("input", () => grow(sinput));
const studio = { kind: "picture", res: "4K", ratio: "16:9", dur: "4s", fps: "30" };
$$(".seg button").forEach((b) => b.addEventListener("click", () => {
  $$(".seg button").forEach((t) => t.classList.toggle("on", t === b));
  studio.kind = b.dataset.kind; const video = studio.kind === "video";
  $("#p-dur").hidden = !video; $("#p-fps").hidden = !video; sinput.placeholder = video ? "describe the shot" : "describe the picture";
}));
$$(".chips").forEach((g) => g.addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  g.querySelectorAll("button").forEach((t) => t.classList.toggle("on", t === b));
  studio[g.id] = b.dataset.v;
  if (g.id === "ratio") $("#frame").dataset.ratio = b.dataset.v;
  $("#frame-size").textContent = frameSize();
}));
function frameSize() {
  const w = studio.res === "4K" ? 3840 : 1920, h = studio.res === "4K" ? 2160 : 1080;
  return studio.ratio === "16:9" ? `${w} × ${h}` : studio.ratio === "9:16" ? `${h} × ${w}` : `${h} × ${h}`;
}
$$(".param input[type=range]").forEach((r) => r.addEventListener("input", () => (r.nextElementSibling.value = r.value)));
sinput.addEventListener("input", () => { $("#render").disabled = !sinput.value.trim(); });
sinput.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); queueJob(); } });
$("#render").addEventListener("click", queueJob);
// the queue lives on the server (results/studio/queue.json), where the model's studio tool and the 4K agent see it too
let jobs = [];
async function studioPost(body) { try { jobs = await (await post("/studio", body)).json(); } catch (e) {} renderJobs(); }
async function loadJobs() { try { jobs = await (await fetch("/studio")).json(); } catch (e) {} renderJobs(); }
function queueJob() {
  const prompt = sinput.value.trim(); if (!prompt) return;
  const job = { prompt, kind: studio.kind, size: frameSize(), dur: studio.kind === "video" ? studio.dur + " · " + studio.fps + "fps" : "", steps: $("#steps").value, guide: $("#guide").value, seed: $("#seed").value.trim() || "random" };
  sinput.value = ""; grow(sinput); $("#render").disabled = true; studioPost({ job });
}
let shown = null;                                                     // which finished render the frame shows
function renderJobs() {
  const q = $("#queue"); q.innerHTML = "";
  if (!jobs.length) q.innerHTML = `<div class="dim small">empty</div>`;
  for (const j of jobs) {
    const el = document.createElement("div"); el.className = "job";
    el.innerHTML = `<div></div><div class="meta">${esc(j.kind)} · ${esc(j.size)}${j.dur ? " · " + esc(j.dur) : ""}${j.by === "model" ? " · <i>from the model</i>" : ""} · ${esc(j.status)}</div><button class="drop" type="button">${j.status === "done" ? "forget" : "remove"}</button>`;
    el.firstElementChild.textContent = j.prompt;
    if (j.status === "done" && j.out) el.firstElementChild.addEventListener("click", () => showRender(j));
    el.querySelector(".drop").addEventListener("click", () => studioPost({ drop: j.id }));
    q.appendChild(el);
  }
  const running = jobs.find((j) => j.status.startsWith("rendering")); const pct = running ? parseInt((running.status.match(/(\d+)%/) || [0, 0])[1], 10) : 0;
  $("#frame-progress").style.width = running ? pct + "%" : "0";
  $("#frame-note").textContent = running ? `${running.status} · ${running.prompt.slice(0, 60)}` : "no model attached yet · the stub agent draws placeholders";
  const done = jobs.filter((j) => j.status === "done" && j.out); const strip = $("#strip"); strip.innerHTML = "";
  for (const j of done.slice().reverse()) {
    const a = document.createElement("a"); a.href = "#"; a.title = j.prompt; a.className = shown && shown.id === j.id ? "on" : "";
    const img = document.createElement("img"); img.src = outUrl(j); img.alt = j.prompt; a.appendChild(img);
    a.addEventListener("click", (e) => { e.preventDefault(); showRender(j); }); strip.appendChild(a);
  }
  if (!shown && done.length) showRender(done[done.length - 1]);
}
const outUrl = (j) => "/out/" + j.out.split("/").pop();            // results/studio/out/<name> is served at /out/<name>
function showRender(j) {
  shown = j; const img = $("#frame-img"); img.src = outUrl(j); img.hidden = false;
  $("#frame-size").textContent = j.size; $$("#strip a").forEach((a) => a.classList.toggle("on", a.title === j.prompt));
}
loadJobs(); setInterval(() => { if ($("#studio").classList.contains("on")) loadJobs(); }, jobs.some((j) => j.status.startsWith("rendering")) ? 1000 : 2500);
