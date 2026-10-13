# The loki client

Two front ends over the engine's line protocol: `chat.py`, a full-screen terminal, and `app.py`, a window
served on `127.0.0.1:8765`. Both share the turn logic, settings, conversations and the tool belt. Everything
here is standard-library Python 3.

```
loki                                  # bin/loki (.cmd on Windows): the terminal on the engine
python3 loki/app.py                   # the window
python3 loki/chat.py --replay 4       # the terminal against three captured turns at 4 tokens/s, no engine
python3 loki/tests.py                 # the headless tests over the replay
```

## The terminal

Bottom up: the input box; the hood while the model thinks (a dim `thinking · 1m 12s` line and the last
`bars` lines of thought); the conversation above. When the thought closes the hood drops over it and leaves
a dim `thought for Ns` line; the answer streams above the box, then a dim stats line. Tokens arrive in bursts
from the engine and are released at the measured rate, so the thought flows evenly.

**Keys.** Enter sends. Tab opens the whole thought over the screen (scroll with Up/Down/PgUp/PgDn or the
wheel) and closes it. Esc stops a running turn, or clears the box. Ctrl-C stops a turn, or quits on a second
press. PgUp/PgDn and the wheel scroll the conversation. Ctrl-J breaks a line; pasted newlines stay in one
message. Up on an empty box recalls what you sent. Ctrl-←/→ by word, Ctrl-K / Ctrl-U kill to the end / start,
Ctrl-W the word before the caret. Enter while the model is answering queues the message. `y` / `n` / `a`
answer a tool that asks.

**Files with a message.** `@notes.md` (or `@"a path with spaces"`) anywhere in a message brings that file's
text along, cut at `attach_cap` characters; Tab completes the path.

**Commands.** `/new` next conversation · `/chats` the last twelve · `/find WORDS` search every conversation ·
`/open N` continue one · `/restart` the engine · `/again` ask the last message again · `/edit [N]` edit the
Nth-last message and continue from it · `/fast` `/deep` presets · `/think N` `/answer N` token budgets
(0 = unlimited) · `/bars N` · `/stats` · `/ctx [N]` · `/sandbox [on|off]` · `/memory` · `/tools` `/tool NAME
allow|ask|off` `/hops N` `/rules` `/cap N` · `/copy` the last answer to the clipboard · `/queue [clear]` ·
`/quit` · `/help`.

Terminals without the box-drawing glyphs get ASCII rules automatically (`--ascii` forces it); `--no-mouse`
leaves the wheel alone. Piped stdin falls back to a plain line mode.

## Settings

`chat_settings.json` in the repository root, created on first run: `ctx` (context positions), `think` and
`answer` (token budgets, 0 = unlimited), `memory` (`shared` keeps the model's memory across conversations,
`sandbox` gives this conversation scratch memory that is deleted at exit), `bars`, `lift`, `stats`, `bell`,
`attach_cap`, the sampling dials `temp`, `top_k`, `top_p`, `seed`, the `stop` list, the tool policy
(`tools`, `tool_policy`, `tool_rules`, `tool_cap`), and `presets`. Transcripts go to `results/chat/`.

## The tool belt

With tools on (`"tools": true`, or `auto` with a context of 2048 or more) the model may end an answer with
one `<tool>{"name": ..., "args": {...}}</tool>` call; the result comes back as the next user message and the
loop continues up to `hops`. Tools: `search`, `fetch`, `clock`, `notes`, `files`, `write`, `shell`, `python`,
`studio`. Each has a policy, `allow`, `ask` or `off`; the client asks in the terminal or the window.

## Protocol

stdin, one command per line: `U <think_budget> <answer_budget> <escaped text>`, `R` (clear the cache),
`Q` (save memories and exit), `X` (stop the current turn). stdout: `READY <ctx>`, `T <id> <piece>`,
`F <id> <piece>` (forced think-close), `END <eos|cap|full|stop> <think_n> <answer_n> <secs> <pos> <ctx>`,
`RESET`, `FULL`. Think close is token 128799, end of sequence is 1. Escaping: `\n` for newline, `\\` for a
backslash.
