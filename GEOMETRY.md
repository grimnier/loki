# The geometry: the shape, carved out

What the engine is doing when it runs from memory, stated as one geometry, with the terms fixed. Everything here is
either measured in this tree (numbers cited) or is the direct reading of those numbers. Oct 9, 2026.

## 1. The space

The weights are one line: 671,026,419,200 numbers, index `i = 0 .. 671,026,419,199`, in the order the file holds
them. That line is the whole space. A position on it is a share and an angle, the same number:

```
P(i) = i / 671,026,419,200          θ(i) = 360° · P(i)          (X, Y) = (cos θ, sin θ)
```

A token id `k` has two fixed homes on the line, its head row and its embedding row, 7,168 numbers each:

```
head row:       i = k · 7168                       (the line opens with the head)
embedding row:  i = 926,686,208 + k · 7168
```

All 129,280 ids, both rows, are 0.2762% of the line: the vocabulary is the first degree of the circle. The other
359 degrees are what a walk is fed through after it. Binary prefixes of an index are arcs: one bit is half the
circle, two bits a quadrant, four bits a 22.5° sector.

## 2. A state

At every layer a walk has a state `x ∈ R^7168` (the normalised input to that layer's experts). Its sketch is a
fixed random projection of the unit state, normalised:

```
u = x / ‖x‖        s = P u  (P: 256 × 7168, entries ±0.0625, seed 0x9E3779B97F4A7C15)        ŝ = s / ‖s‖
```

Nearness between two states is their cosine, which is an angle: `sim = ŝ_a · ŝ_b`, `angle = arccos(sim)`.
0.999 is 2.6°, 0.99 is 8.1°, 0.98 is 11.5°, 0.95 is 18.2°.

A stored state carries, beside its sketch: the push its experts gave (`out`, 7,168 numbers, the stored output),
the experts it routed to (`route[0..1]`, each an angle on the line), the next 16 token ids of its walk
(`nxt[0..15]`, each an angle on the vocab arc), and the discrete probabilities the head gave at its top when the
walk reached it. A state therefore has coordinates in both senses at once: where on the line it was computed,
and where on the vocab circle it points.

## 3. The walk

The model is a deterministic map on states, layer by layer and token by token:

```
x_{i+1} = f(x_i)
```

The same `f` for every question. A question is a starting point; a walk is the orbit of that point under `f`.
Known states are points on orbits that have been walked and stored. Unknown states are points not yet near a
stored one.

## 4. Collision

A walk that comes within the working angle of a stored state has collided with a known orbit. From the
collision on, the known orbit is the walk: the stored push stands in for the layer, the stored continuation
stands in for the tokens, and nothing on the line is read. That is the whole mechanism; the engine's 12 to 15
tokens per second on known ground is this and nothing else.

This is Pollard's rho with distinguished points, term for term: `f` is the iterated map, the stored states are the
distinguished points, the table is the distinguished-point store, a hit at the working angle is the collision,
the stored tail is the merge, and the expected walk to a collision grows with the square root of the effective
space, not with its size. Every orbit stored shortens the next walk to a collision.

Measured, on a question the memory had never seen ("How does a bicycle stay upright when it moves?", 240 stored
questions, copies, the four runs):

| working angle | states served from the store | passes with no read at all | speed | words |
|---|---|---|---|---|
| 2.6° | 9% | 2 | 0.43 tok/s | `...I remember that once when I was a kid, I tried to` |
| 8.1° | 48% | 170 | 1.00 tok/s | `...I remember that once a bicycle is moving, even if it's` |
| 11.5° | 73% | 178 | 1.91 tok/s | `...I remember that once when I was a kid, I tried to` |
| 18.2° | 53% | 156 | 1.25 tok/s | `...I remember that if you try to ride a bike really slowly,` |

At 11.5° the prefill of the unseen question read nothing from the line. The 2.6° row is the bar the engine runs
in production; it was chosen for bit-identical replay, and it is what made the brute force look necessary.

## 5. Groups and subgroups

Orbits merge. Where many walks run along the same states, those states form a group; where the group splits, the
branches are its subgroups. The four runs above show one of each in the open:

```
group:      Okay, so I need to figure out how a bicycle stays upright when it's moving.
            Let me start by recalling what I know. I remember that
subgroups:  | once when I was a kid, I tried to
            | once a bicycle is moving, even if it's
            | if you try to ride a bike really slowly,
```

The largest group is the voice. The model thinks in the first person, and every walk starts there: BOS, the
role marks, `<think>`, `Okay, so I need to figure out`, `Let me start by recalling what I know`. The role marks
sit on the seam of the vocab circle (ids 0, 1 and 128799..128804: 0° and 353.8° in the 2^17 space), and every
orbit passes through them on the way in and on the way out. Templates inside a thought (`the basic answer`,
`But wait, maybe I should`) are groups; the close (`</think>`, the answer frame, EOS) is a group.

A family, in the engine's current code, is a subgroup found by search: the 8 nearest stored states within 53°
that carry a continuation, voted on by closeness. It is recomputed every pass and discarded.

## 6. The question as a point

`<question>` is a known state from the moment it is read: its frame, its role marks, its shape all collide
with the voice and the templates. What is left is the content, and the content is itself a state on some orbit
(the orbit of every walk that ever carried that content, or a product of orbits that carried its parts). It is
found by collision like everything else; there is no slot that must be computed from the line by brute force.
How much of a fresh question falls through to a read is the only cold term, and it shrinks as the square of the
store, which is the birthday curve, not a floor.

## 7. What the engine does today, against this shape

| the shape | the engine now |
|---|---|
| distinguished points are kept | 4,096 per layer, oldest unused evicted: they roll |
| a collision at the working angle serves | serves only at 2.6° (`--geo-sim 0.999`) |
| any stored state can serve | only states that came from a read may serve; served states are "derived" and only draft |
| groups and subgroups are held | re-found by nearest-neighbour search every pass, then dropped |
| the push, the route, the tokens, the probabilities are one record | push + route + next tokens stored; the head's probabilities are logged, not stored beside the state |
| the walk continues on the known orbit | continues the stored tail; stores the new states of its own orbit |

The five rows on the right are choices, not hardware. The bytes that a cold pass reads are the cost of those
choices.

## 8. The patch, in order

1. **Freeze.** Distinguished points become fixed: no eviction of a state that has served or been merged into;
   a state that was served may serve. The table grows; the store is the orbit history.
2. **The working angle.** The serving bar becomes the dial it is (11.5° is where the unseen question ran clean);
   derived-only becomes a flag, off.
3. **Groups as records.** When walks merge, the merged span is written once as a group with its branch list and
   the probabilities at each branch point; a collision with a group serves the whole span, and the branch is
   chosen from the head's discrete distribution at the point, not by a per-pass vote.
4. **Reads only until collision.** A walk reads the line only while it is on no known orbit; the first collision
   ends the reading for that walk. The counter to watch is the fraction of a fresh question's row-layers that
   never collide, as a function of the store's size.
5. **The curve.** Fresh questions in a row, at the working angle, with the store frozen and growing: coverage
   against stored states. That is the birthday curve, and it is the measurement that says whether the cold term
   vanishes the way the geometry says it does.

Files: the engine's lookup and store are in `src/server_core_async.cpp` (the `nearest stored state` block, the
family vote, `gmake`/`gout_read`), the production build is `patch_fused.py` → `src/server_core_fused.exe`,
the memory is `ptx_trees/geo_memory.bin` with its `.kv/.out/.out32` sidecars and `route_memory.bin`.
