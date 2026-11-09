"""Step 3 (pure code): turn rough LLM/heuristic timestamps into valid, clean-cut promos."""
from __future__ import annotations
from dataclasses import dataclass, field
from .parser import Episode, fmt, transcript

MIN_LEN, MAX_LEN = 20.0, 60.0


@dataclass
class Seg:
    ep: int
    start: float
    end: float

    @property
    def dur(self):
        return self.end - self.start


@dataclass
class Cand:
    cid: str
    segs: list
    source: str                     # llm | heuristic | stitched
    hook: str = ""
    hook_quote: str = ""
    cliff: str = ""
    why: str = ""
    prior_hook: float | None = None   # scores claimed by pass-A (before the independent judge)
    prior_cliff: float | None = None
    flags: list = field(default_factory=list)
    judge: dict = field(default_factory=dict)
    score: float = 0.0
    rank: int | None = None

    @property
    def dur(self):
        return sum(s.dur for s in self.segs)

    @property
    def episodes(self):
        return sorted({s.ep for s in self.segs})


class EpisodeIndex:
    """Knows where it is safe to cut an episode."""

    def __init__(self, ep: Episode):
        self.ep = ep
        cams = [e for e in ep.events if e.kind == "camera"]
        base = cams or ep.content_events
        self.shot_starts = sorted({e.start for e in base} | {0.0})
        self.shot_ends = sorted({e.end for e in base} | {ep.duration})
        self.dlg = [e for e in ep.events if e.kind == "dialogue"]

    def inside_dialogue(self, t: float):
        for d in self.dlg:
            if d.start + 0.05 < t < d.end - 0.05:
                return d
        return None

    def snap_start(self, s: float) -> float:
        s = max(0.0, min(s, self.ep.duration))
        d = self.inside_dialogue(s)
        if d:
            s = d.start
        c = [x for x in self.shot_starts if x <= s + 1e-6 and s - x <= 2.0 and not self.inside_dialogue(x)]
        return max(c) if c else s

    def snap_end(self, e: float) -> float:
        e = max(0.0, min(e, self.ep.duration))
        d = self.inside_dialogue(e)
        if d:
            e = d.end
        c = [x for x in self.shot_ends if x >= e - 1e-6 and x - e <= 2.5 and not self.inside_dialogue(x)]
        return min(c) if c else e

    def clean_starts_after(self, t):
        return [x for x in self.shot_starts if x > t and not self.inside_dialogue(x)]

    def clean_starts_before(self, t):
        return [x for x in self.shot_starts if x <= t and not self.inside_dialogue(x)]


def build_indexes(episodes):
    return {e.num: EpisodeIndex(e) for e in episodes}


def normalise(cand: Cand, idx: dict):
    """Snap every segment to clean cut points, merge overlaps, force 20-60s total.
    Returns (ok, reason). Mutates cand."""
    segs = []
    for s in cand.segs:
        ix = idx.get(s.ep)
        if ix is None:
            return False, f"unknown episode {s.ep}"
        if not (0 <= s.start < s.end <= ix.ep.duration + 1.0):
            return False, f"timestamps outside episode {s.ep} ({fmt(s.start)}-{fmt(s.end)}, len {fmt(ix.ep.duration)})"
        segs.append(Seg(s.ep, ix.snap_start(s.start), ix.snap_end(s.end)))
    merged = []
    for s in segs:                       # merge touching/overlapping segments of the same episode
        if merged and merged[-1].ep == s.ep and s.start <= merged[-1].end + 0.3:
            merged[-1].end = max(merged[-1].end, s.end)
        else:
            merged.append(s)
    if len(merged) > 3:
        return False, "more than 3 segments"
    cand.segs = merged
    total, first = cand.dur, cand.segs[0]
    ix = idx[first.ep]
    if total > MAX_LEN:                   # trim setup from the front, keep the cliffhanger ending
        need = first.start + (total - MAX_LEN)
        opts = [x for x in ix.clean_starts_after(first.start) if x >= need - 1e-6 and x < first.end - 1.0]
        if not opts:
            return False, f"too long ({total:.1f}s) and no clean trim point"
        first.start = min(opts)
        cand.flags.append("trimmed_to_fit")
    elif total < MIN_LEN:                 # add setup before the hook
        need = first.start - (MIN_LEN - total)
        opts = ix.clean_starts_before(need)
        first.start = max(opts) if opts else 0.0
        cand.flags.append("extended_to_fit")
        if cand.dur < MIN_LEN:
            last = cand.segs[-1]
            ie = idx[last.ep]
            ends = [x for x in ie.shot_ends if x >= last.end + (MIN_LEN - cand.dur) - 1e-6 and not ie.inside_dialogue(x)]
            if not ends:
                return False, f"too short ({cand.dur:.1f}s) and cannot extend"
            last.end = min(ends)
            cand.flags.append("end_extended")
    if not (MIN_LEN - 0.01 <= cand.dur <= MAX_LEN + 0.01):
        return False, f"duration {cand.dur:.1f}s outside 20-60s after repair"
    return True, "ok"


def iou(a: Cand, b: Cand) -> float:
    """Overlap between two candidates, as share of the shorter one (same-episode footage only)."""
    ov = 0.0
    for x in a.segs:
        for y in b.segs:
            if x.ep == y.ep:
                ov += max(0.0, min(x.end, y.end) - max(x.start, y.start))
    return ov / max(1e-6, min(a.dur, b.dur))


def clip_text(cand: Cand, episodes_by_num: dict) -> str:
    """What a viewer of this exact clip would see/hear, with times relative to each segment start."""
    out = []
    for i, s in enumerate(cand.segs, 1):
        body = transcript(episodes_by_num[s.ep], rel_to=s.start, lo=s.start, hi=s.end)
        out.append(f"--- Segment {i} (from Episode {s.ep}, {s.dur:.0f}s) ---\n{body}")
    return "\n".join(out)


def seg_str(s: Seg) -> str:
    return f"E{s.ep} {fmt(s.start)}-{fmt(s.end)}"
