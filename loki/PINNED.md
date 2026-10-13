# Pinned for the owner's review — do not overwrite, merge over, or delete

The state of the client at commit `253766d` is kept as branch `pin/loki-2026-10-08` on GitHub. It is the owner's
to review; nothing on it is to be changed, rebased, squashed or deleted by any session until he has gone over it
and said what stays. The working branch `loki-hood-and-window` carries on from it, and the only work allowed on
that branch until further notice is **user-interface, terminal and visual design** — not the parts below.

## What is frozen (the conversation and dialogue side)

Everything about how a conversation is held, carried, continued, branched and found:

- `chat.py`: `submit`, `send_turn`, `pump`, `finish`, `continue_turn`, the context rollover and carry
  (`carry_next`, `history`), the conversation files (`save_conv`, `new_conv`, `open_conv`, `list_convs`,
  `find_convs`), `branch_at`, `/again`, `/edit`, `/find`, `/open`, `/chats`, `/new`, the type-ahead `queue`,
  `attach` (@file), the tool loop (`run_tool`, `answer_tool`, `poll_tool`, `abort_tool`, `engine_died`,
  `/restart`), the transcript files, the sandbox / shared memory modes merged from `main`.
- `tools/`: the belt, its policies, rules and preamble.
- `app.py`: the session, the delta stream, every endpoint.
- `studio/`: the queue, the runner, the agent protocol.
- The replay turns in `REPLAY`.

These came from this session's suggestions. The owner has his own direction on dialogue (his `main`, the fixtures
and `CLOUD_WORK.md`) and wants to compare the two before anything bleeds into the other.

## What is open

The look and feel, in both clients: typography, spacing, tones, the hood's motion, the box, the palette and
queue rows, the rail, knobs, chats and studio pages, compact mode, the terminal's drawing (`term.py`), glyphs,
widths, easing, the frame clock — and what the interface *shows* about the engine's pace once the three kinds of
token in `fixtures/` are read (geometry runs, verified drafts, new content). Anything found wrong in the frozen
parts during that work is to be reported, not fixed.

## Known and not fixed (reported 2026-10-08)

- After a tool call the stripped `<tool>…</tool>` text can come back into the answer line while the tool runs or
  asks: `finish` strips it from the log entry but `smooth()` keeps re-applying `t["ans"]`, which still holds it.
  The fix is one line in `finish` (`t["ans"] = shown` after the strip) — in the frozen part, so not made.
