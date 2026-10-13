# tests.py -- the interface, checked headless: scripted keys against the replay, every frame captured, asserted on.
#   python3 loki/tests.py            runs all (about two minutes; the replay runs at 30 tokens/s)
#   python3 loki/tests.py -k hood    runs the ones whose name contains "hood"
#   Each test runs chat.py's TUI in-process with a 4096 context so the tool belt is on, and writes nothing outside
#   results/ (conversations, transcripts and tool logs it makes are removed after).
import os, sys, re, glob, shutil, time, unittest, tempfile
LOKI = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(LOKI); sys.path.insert(0, LOKI)
import chat
START = time.time()

def run(keys, size=(100, 24), replay=30, cfg=None):
    """Returns the frames (list of strings) and the TUI after the script ran."""
    c = dict(chat.DEFAULTS); c.update({"ctx": 4096}); c.update(cfg or {})
    chat.save_settings = lambda cfg: None                              # never touch chat_settings.json from a test
    if os.path.exists(chat.DRAFT): os.remove(chat.DRAFT)               # a draft left by the last test would land in this one's box
    fd, path = tempfile.mkstemp(suffix=".txt"); os.close(fd)
    tui = chat.TUI(c, chat.Replay(c, replay), chat.ScriptedKeys(keys), chat.Screen(path, size)); tui.run()
    with open(path, encoding="utf-8") as f: frames = f.read().split("---- frame ")[1:]
    os.remove(path)
    return [re.sub(r"\x1b\[[0-9;]*m", "", f) for f in frames], tui

class Hood(unittest.TestCase):
    def test_hood_shows_four_lines_and_opens(self):
        frames, _ = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:1.8><TAB><WAIT:0.5><TAB><WAIT:7><QUIT>")
        thinking = [f for f in frames if "thinking ·" in f]
        self.assertTrue(thinking, "no hood frames")
        rows = lambda f: [r for r in f.split("\n")[1:] if r.startswith("  ") and r.strip() and "thinking" not in r and "›" not in r]
        self.assertTrue(all(len(rows(f)) <= 4 for f in thinking if "   " not in f[:4]), "more than four lines under the hood")
        self.assertTrue(any(len([r for r in f.split("\n") if r.startswith("  ")]) > 6 for f in thinking), "Tab did not open the hood")
        self.assertIn("thought for", frames[-1]); self.assertIn("tok/s", frames[-1]); self.assertNotIn("thinking ·", frames[-1])
    def test_text_flows_without_reflow(self):
        frames, _ = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:6><QUIT>", size=(60, 12))
        prev = None
        for f in [f for f in frames if "thinking ·" in f]:
            rows = [r.rstrip() for r in f.split("\n")[1:] if r.startswith("  ") and "thinking" not in r and r.strip()]
            if prev:                                                   # every finished row is still there (or has scrolled off the top); only its clipped last word may move down
                stems = [a.rsplit(" ", 1)[0] for a in prev[:-1] if a.rsplit(" ", 1)[0].strip()]
                kept = [i for i, s in enumerate(stems) if any(b.startswith(s) for b in rows)]
                if kept: self.assertEqual(kept, list(range(kept[0], len(stems))), f"a row reflowed between frames: {stems} -> {rows}")
            prev = rows
    def test_the_hood_is_a_tv(self):   # the owner's rule: the TV is on while the thought generates; when it ends the chain is gone from the screen and "thought for Ns" is all that remains. Nothing brings it back. Not to be eroded.
        frames, tui = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:9><TAB><WAIT:0.5><TAB><WAIT:0.5><QUIT>", size=(100, 20))
        TH = ("Okay, the user is asking", "Rayleigh scattering", "Let me answer now"); seen = lambda f: any(t in f for t in TH)
        live = [f for f in frames if "thinking" in f]; self.assertTrue(any(seen(f) for f in live), "the thought never showed while it generated")
        last_live = max(i for i, f in enumerate(frames) if "thinking" in f); after = frames[last_live + 1:]; self.assertTrue(after)
        self.assertTrue(not any(seen(f) for f in after), "thought text came back on screen after the hood dropped (Tab was pressed twice)")
        self.assertIn("thought for", after[0]); self.assertIn("thought for", after[-1])
        self.assertTrue(any(e.get("thought") for e in tui.log), "the thought text must still be kept in the log, just not drawn")

class Keys(unittest.TestCase):
    def test_ctrl_c_needs_two_presses(self):
        frames, tui = run("<WAIT:0.3><CTRLC><WAIT:0.3><QUIT>")
        self.assertTrue(any("Ctrl-C again to quit" in f for f in frames))
    def test_esc_stops_a_turn(self):
        frames, _ = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:2><ESC><WAIT:3><QUIT>")
        self.assertTrue(any("stopped" in f for f in frames[-3:]))
    def test_scroll_hint_while_streaming(self):
        frames, _ = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:2.5><WHEELUP><WAIT:0.3><QUIT>", size=(60, 8))   # while the turn is still running
        self.assertTrue(any(re.search(r"↓ \d more", f) for f in frames), "no scroll hint")
        self.assertTrue(any("▏" in f for f in frames), "no hairline while scrolled")

class Tools(unittest.TestCase):
    def test_tool_turn_runs_and_shows_one_line(self):
        frames, tui = run("<WAIT:0.5>What time is it?<ENTER><WAIT:9><QUIT>")
        last = frames[-1]
        self.assertIn("→ clock", last); self.assertIn("1 tool call", last); self.assertNotIn("<tool>", last)
        self.assertTrue(any("→ clock · 0s\n" in f and "thinking" not in f for f in frames), "the hood never showed the tool running")
    def test_ask_then_deny(self):
        frames, _ = run("<WAIT:0.5>What is in this folder?<ENTER><WAIT:6><n><WAIT:4><QUIT>")
        self.assertTrue(any("y / n / a" in f for f in frames)); self.assertIn("declined", frames[-1])
    def test_always_remembers(self):
        frames, tui = run("<WAIT:0.5>What is in this folder?<ENTER><WAIT:6><a><WAIT:4>What is in this folder?<ENTER><WAIT:9><QUIT>")
        asks = sum(1 for a, b in zip([""] + frames, frames) if "y / n / a" in b and "y / n / a" not in a)   # episodes, not frames: the mark pulses while it waits
        self.assertEqual(asks, 1, "asked twice")
        self.assertEqual(tui.cfg["tool_rules"], [{"tool": "shell", "match": "ls", "policy": "allow"}])
    def test_tools_off_below_2048(self):
        frames, _ = run("<WAIT:0.5>What time is it?<ENTER><WAIT:9><QUIT>", cfg={"ctx": 256})
        self.assertNotIn("→ clock", frames[-1])

class Typing(unittest.TestCase):
    def test_palette_completes_and_runs(self):
        frames, tui = run("<WAIT:0.3>/th<WAIT:0.2><TAB>60<ENTER><WAIT:0.3><QUIT>")
        self.assertTrue(any("/think N" in f and "think tokens" in f for f in frames), "no palette")
        self.assertEqual(tui.cfg["think"], 60); self.assertIn("think 60", frames[-1])
    def test_enter_mid_turn_queues_and_sends(self):
        frames, tui = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:1.5>And at sunset?<ENTER><WAIT:14><QUIT>")
        self.assertTrue(any("queued" in f for f in frames), "nothing queued")
        self.assertIn("And at sunset?", frames[-1]); self.assertEqual(sum(1 for e in tui.log if e["kind"] == "user"), 2)
    def test_up_recalls_and_draft_survives(self):
        frames, tui = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:8><UP><WAIT:0.2><QUIT>")
        self.assertEqual(tui.inp, "What color is the sky?")
        self.assertEqual(open(chat.DRAFT, encoding="utf-8").read(), "What color is the sky?"); os.remove(chat.DRAFT)
    def test_word_jumps_and_kill(self):
        _, tui = run("<WAIT:0.3>one two three<CTRLLEFT><CTRLLEFT><CTRLK><WAIT:0.1><QUIT>")
        self.assertEqual(tui.inp, "one ")
    def test_wide_characters_keep_the_border(self):
        frames, _ = run("<WAIT:0.3>日本語 🙂 café<WAIT:0.2><QUIT>", size=(50, 10))
        row = next(r for r in frames[-1].split("\n") if "café" in r)
        self.assertEqual(chat.dw(row), 49, "the box row must be as wide as its corner rows"); self.assertTrue(row.endswith("│"))

class Pace(unittest.TestCase):
    def test_hood_shows_the_live_rate_and_the_summary_after(self):
        frames, _ = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:9><QUIT>")
        heads = [re.search(r"thinking[^\n]*", f).group(0) for f in frames if "thinking" in f]
        self.assertTrue(any("tok/s" in h for h in heads), "no live rate in the hood"); self.assertTrue(any("stored" in h or "drafted" in h for h in heads), "no pass kind in the hood")
        self.assertRegex(frames[-1], r"\d+\.\d tok/s · \d+ passes · \d+ tokens from drafts · \d+ from stored runs · \d+ cold")   # the turn added up by the client; the engine writes no per-turn summary in chat mode

class Motion(unittest.TestCase):
    def test_text_flows_evenly_between_bursts(self):
        frames, _ = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:6><QUIT>", size=(80, 16), replay=2)
        texts = []
        for f in [f for f in frames if "thinking" in f]:
            texts.append(" ".join(r.rstrip() for r in f.split("\n")[1:] if r.startswith("  ") and "thinking" not in r and r.strip()))
        changed = sum(1 for a, b in zip(texts, texts[1:]) if a != b)
        self.assertGreater(changed, 25, f"the thought moved in only {changed} frames: it is arriving in bursts, not flowing")
    def test_hood_closes_on_a_curve(self):
        frames, _ = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:8><QUIT>")
        counts = []
        for f in [f for f in frames if "thinking" in f][-12:]:
            counts.append(len([r for r in f.split("\n")[1:] if r.startswith("  ") and "thinking" not in r and r.strip()]))
        self.assertTrue(any(0 < c < 4 for c in counts), f"no intermediate hood heights while closing: {counts}")

class Branches(unittest.TestCase):
    def test_again_keeps_a_branch(self):
        frames, tui = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:8>/again<ENTER><WAIT:9><QUIT>")
        self.assertEqual(sum(1 for e in tui.log if e["kind"] == "user"), 1); self.assertEqual(len(tui.branches), 1)
        self.assertTrue(any(e["kind"] == "answer" for e in tui.branches[0]["log"]))
    def test_edit_loads_the_message(self):
        frames, tui = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:8>/edit<ENTER><WAIT:0.3><QUIT>")
        self.assertEqual(tui.inp, "What color is the sky?"); self.assertEqual(tui.log, []); self.assertEqual(len(tui.branches), 1)
    def test_presets(self):
        _, tui = run("<WAIT:0.3>/fast<ENTER><WAIT:0.3><QUIT>")
        self.assertEqual((tui.cfg["ctx"], tui.cfg["think"], tui.cfg["answer"], tui.cfg["tools"]), (128, 60, 80, False))

class Attach(unittest.TestCase):
    def test_at_file_goes_with_the_message(self):
        path = os.path.join(HERE, "results", "attach_test.md"); open(path, "w", encoding="utf-8").write("kelp forests grow fast\n")
        try:
            frames, tui = run("<WAIT:0.5>What color is the sky? @results/attach_t<TAB><ENTER><WAIT:9><QUIT>")
            self.assertIn("attached attach_test.md", frames[-1]); self.assertIn("@results/attach_test.md", frames[-1])
            self.assertEqual(tui.history[-1][0], "What color is the sky? @results/attach_test.md", "the carried message must be what was typed, not the file")
        finally: os.remove(path)
    def test_missing_file_is_said(self):
        frames, _ = run("<WAIT:0.5>What color is the sky? @nope.txt<ENTER><WAIT:9><QUIT>")
        self.assertIn("no file @nope.txt", frames[-1])

class Finding(unittest.TestCase):
    def test_find_then_open(self):
        frames, tui = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:8>/new<ENTER>Why do we have eyebrows?<ENTER><WAIT:9>/find overcast<ENTER><WAIT:0.3>/open 1<ENTER><WAIT:0.3><QUIT>")
        self.assertTrue(any("1. " in f and "overcast" in f and "/open N" in f for f in frames), "no find result")
        self.assertEqual(tui.conv["title"], "What color is the sky?")

class Signals(unittest.TestCase):
    def test_bell_when_an_answer_took_a_while(self):
        frames, _ = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:8><QUIT>", cfg={"bell_after": 1})
        self.assertTrue(any("---- bell" in f for f in frames), "no bell in the frames file")
    def test_engine_death_and_restart(self):
        c = dict(chat.DEFAULTS); c["ctx"] = 4096; chat.save_settings = lambda cfg: None
        fd, path = tempfile.mkstemp(suffix=".txt"); os.close(fd)
        tui = chat.TUI(c, chat.Replay(c, 30), chat.ScriptedKeys("<WAIT:0.5>/restart<ENTER><WAIT:2>What color is the sky?<ENTER><WAIT:9><QUIT>"), chat.Screen(path, (100, 24)))
        tui.eng.q.put(None)                                            # the engine goes away before the first key
        tui.run()
        with open(path, encoding="utf-8") as f: frames = f.read().split("---- frame ")[1:]
        os.remove(path)
        self.assertTrue(any("engine stopped" in f and "/restart" in f for f in frames), "death not announced")
        self.assertIn("The color of the sky", frames[-1]); self.assertFalse(tui.dead)

class Conversations(unittest.TestCase):
    def test_new_saves_and_open_continues(self):
        frames, tui = run("<WAIT:0.5>What color is the sky?<ENTER><WAIT:8>/new<ENTER>Why do we have eyebrows?<ENTER><WAIT:9>/open 2<ENTER><WAIT:0.3>And at sunset?<ENTER><WAIT:9><QUIT>")
        last = frames[-1]
        self.assertIn("What color is the sky?", last); self.assertIn("And at sunset?", last); self.assertNotIn("eyebrows", last)
        self.assertIn("continuing", last); self.assertEqual(tui.conv["turns"], 2)

def cleanup():
    for d in ("conversations", "tools"): shutil.rmtree(os.path.join(HERE, "results", d), ignore_errors=True)
    for f in glob.glob(os.path.join(HERE, "results", "chat", "*.md")):
        try:
            if os.path.getmtime(f) > START: os.remove(f)
        except Exception: pass

if __name__ == "__main__":
    try: unittest.main(verbosity=2)
    finally: cleanup()
