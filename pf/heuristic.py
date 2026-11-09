"""Rule-based scorer. Used (a) as a no-API baseline, (b) as an extra candidate source so one bad LLM
reply never leaves an episode uncovered. Deliberately simple and explainable."""
from __future__ import annotations
import re
from .clips import Cand, Seg, MIN_LEN, MAX_LEN

HOT = ("slap", "divorce", "pregnan", "dead", "die", "kill", "murder", "traitor", "betray", "liar", "lie", "secret",
       "truth", "poison", "blood", "gun", "sword", "scream", "how dare", "get out", "guard", "arrest", "execut",
       "prison", "dungeon", "kneel", "beg", "confess", "fired", "disgrace", "humiliat", "mistress", "bastard",
       "fake", "impost", "abandon", "curse", "shame", "scandal", "forbidden", "reveal", "gasp", "stunned", "shock",
       "freez", "collaps", "shove", "grab", "throw", "slam", "shatter", "strike", "faint", "burst", "draws",
       "drag", "never", "regret", "mine", "wife", "husband", "duke", "marry", "married", "baby", "child", "who are you")
TONE_HOT = ("shock", "furious", "menac", "threat", "venom", "tremb", "desperate", "gasp", "scath", "cruel", "savage",
            "cold", "seething", "horrified", "icy", "enraged", "panick")
CLIFF_ACTION = ("door", "enters", "steps in", "appears", "freez", "stares", "gasp", "stunned", "reveal", "holds up",
                "pulls out", "points", "unsheath", "turns to", "lifts", "slowly")


def _hits(text: str, words) -> int:
    t = text.lower()
    return sum(1 for w in words if w in t)


def event_hook_score(ev) -> float:
    if ev.kind == "sfx":
        return 0.0
    t = ev.text
    if ev.kind == "dialogue":
        s = 2.0 + 2.0 * min(2, _hits(t, HOT))
        s += 2.0 if t.rstrip().endswith("?") else 0
        s += 1.5 if "!" in t else 0
        s += 1.0 if len(t.split()) <= 7 else 0
        s += 1.0 if _hits(ev.tone, TONE_HOT) else 0
    else:
        s = 1.5 + 2.5 * min(2, _hits(t, HOT))
    return min(10.0, s)


def event_cliff_score(ev) -> float:
    t = ev.text.strip()
    s = 2.0
    if ev.kind == "dialogue":
        s += 3.0 if t.endswith("?") else 0
        s += 2.0 if t.endswith(("...", "\u2026", "-", "\u2014")) else 0
        s += 1.0 if "!" in t else 0
        s += 1.0 if _hits(ev.tone, TONE_HOT) else 0
        s += 1.5 if _hits(t, HOT) else 0
    else:
        s += 2.0 if _hits(t, CLIFF_ACTION) else 0
        s += 1.5 if _hits(t, HOT) else 0
    return min(10.0, s)


def window_scores(ep, ix, s, e):
    content = ep.content_events
    first3 = [x for x in ep.speech_events if s - 1e-6 <= x.start <= s + 3.0]
    hook = max([event_hook_score(x) for x in first3], default=0.0)
    if not first3:
        hook = 0.0
    last = [x for x in content if e - 3.0 <= x.end <= e + 0.05]
    cliff = max([event_cliff_score(x) for x in last], default=1.0)
    if e >= ep.duration - 1.0:
        cliff = min(10.0, cliff + 3.0)         # episode endings are built as cliffhangers
    inside = sorted((event_hook_score(x) for x in ep.speech_events if x.start >= s and x.end <= e), reverse=True)[:5]
    dens = sum(inside) / max(1, len(inside)) if inside else 0.0
    return hook, cliff, dens


def heuristic_candidates(ep, ix, k=2, prefix="H"):
    starts = [x for x in ix.shot_starts if not ix.inside_dialogue(x)]
    ends = [x for x in ix.shot_ends if not ix.inside_dialogue(x)]
    scored = []
    for s in starts:
        best = None
        for e in ends:
            d = e - s
            if MIN_LEN <= d <= MAX_LEN:
                h, c, dn = window_scores(ep, ix, s, e)
                tot = 0.45 * h + 0.40 * c + 0.15 * dn
                if best is None or tot > best[0]:
                    best = (tot, s, e, h, c)
        if best:
            scored.append(best)
    scored.sort(reverse=True)
    picked = []
    for tot, s, e, h, c in scored:
        if all(e <= p[1] or s >= p[2] for p in picked):
            picked.append((tot, s, e, h, c))
        if len(picked) >= k:
            break
    out = []
    for n, (tot, s, e, h, c) in enumerate(picked, 1):
        first = next((x for x in ep.speech_events if x.start >= s - 1e-6), None)
        last = [x for x in ep.speech_events if x.end <= e + 0.05][-1:]
        hook_txt = (f'{first.speaker}: "{first.text}"' if first and first.kind == "dialogue" else (first.text if first else ""))
        cl = last[0] if last else None
        cl_txt = (f'{cl.speaker}: "{cl.text}"' if cl and cl.kind == "dialogue" else (cl.text if cl else ""))
        c_ = Cand(f"{prefix}{ep.num}.{n}", [Seg(ep.num, s, e)], "heuristic", hook=hook_txt[:200],
                  cliff=cl_txt[:200], why="Rule-based pick: strong opener keywords + unresolved/emphatic ending.",
                  prior_hook=h, prior_cliff=c)
        out.append(c_)
    return out
