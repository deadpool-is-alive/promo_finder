"""Step 1 (pure code, no AI): screenplay .txt -> structured episodes with timestamps."""
from __future__ import annotations
import io, os, re, zipfile
from dataclasses import dataclass, field
from pathlib import Path


def to_sec(v) -> float:
    """Accepts 12.5, '12.5', '00:12.50', '1:02:03.5' -> seconds."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().strip("()[]")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = s.split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"bad timestamp: {v!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def fmt(t: float) -> str:
    h = int(round(t * 100))
    return f"{h // 6000:02d}:{(h % 6000) / 100:05.2f}"


@dataclass
class Event:
    kind: str          # camera | action | dialogue
    start: float
    end: float
    text: str
    speaker: str = ""
    tone: str = ""


@dataclass
class Scene:
    n: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    num: int
    logline: str
    continuity: str
    events: list
    scenes: list
    duration: float
    source: str = ""

    @property
    def content_events(self):          # everything an audience hears/sees except camera notes
        return [e for e in self.events if e.kind != "camera"]

    @property
    def speech_events(self):           # dialogue + on-screen action (not sound cues)
        return [e for e in self.events if e.kind in ("dialogue", "action")]


BLOCK = re.compile(
    r"^\*\*(?P<label>[^*\n]+?):\*\*[ \t]*\n[ \t]*Start:[ \t]*\((?P<s>[\d:.]+)\)"
    r"(?:[ \t]*\|[ \t]*End:[ \t]*\((?P<e>[\d:.]+)\))?[ \t]*\n(?P<body>.*?)"
    r"(?=^\*\*[^*\n]+?:\*\*[ \t]*\n|^SCENE\s+\d+|^--- SCRIPT END|\Z)",
    re.M | re.S | re.I,
)
SCENE = re.compile(
    r"^SCENE\s+(\d+)\s*\n\s*Scene heading:\s*(.+?)\s*\n\s*Scene start:\s*([\d:.]+)\s*\n\s*Scene end:\s*([\d:.]+)",
    re.M | re.I,
)


def parse_episode(text: str, source: str = "") -> Episode:
    m = re.search(r"EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", text, re.I)
    if m:
        num = int(m.group(1))
    else:
        m2 = re.search(r"episode[-_ ]?0*(\d+)", source, re.I)
        num = int(m2.group(1)) if m2 else 0
    g = lambda key: (re.search(rf"^{key}:\s*(.+)$", text, re.M | re.I) or [None, ""])[1].strip()
    events = []
    for b in BLOCK.finditer(text):
        label, body = b["label"].strip(), b["body"].strip()
        s = to_sec(b["s"])
        e = to_sec(b["e"]) if b["e"] else None
        up = label.upper()
        if up == "CAMERA":
            ev = Event("camera", s, e if e is not None else s, body)
        elif up == "ACTION":
            ev = Event("action", s, e if e is not None else s, body)
        elif up in ("SOUND FX", "SFX", "SOUND EFFECT", "SOUND EFFECTS", "MUSIC", "SOUND"):
            ev = Event("sfx", s, e if e is not None else s, body.strip().strip('"\u201c\u201d').strip(), up)
        else:
            sp = re.match(r"(.*?)\s*(?:\((.*)\))?$", label)
            body = body.strip().strip('"\u201c\u201d').strip()
            ev = Event("dialogue", s, e if e is not None else s, body, sp.group(1).strip(), (sp.group(2) or "").strip())
        ev._open_end = e is None
        events.append(ev)
    events.sort(key=lambda x: (x.start, x.end))
    for i, ev in enumerate(events):          # repair missing End values
        if getattr(ev, "_open_end", False):
            nxt = next((x.start for x in events[i + 1:] if x.start > ev.start), None)
            ev.end = nxt if nxt else ev.start + 1.5
    scenes = [Scene(int(a), h, to_sec(s), to_sec(e)) for a, h, s, e in SCENE.findall(text)]
    duration = max([e.end for e in events] + [s.end for s in scenes] + [0.0])
    return Episode(num, g("EPISODE LOGLINE"), g("CONTINUITY NOTES"), events, scenes, duration, source)


def load_episodes(path: str | os.PathLike):
    """Load from a .zip, a folder, or a single .txt. Returns (show_title, [Episode...])."""
    p = Path(path)
    items, title = [], p.stem
    if p.is_file() and p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.lower().endswith(".txt") and "__MACOSX" not in n and not n.endswith("/"):
                    items.append((n, z.read(n).decode("utf-8", "replace")))
                    title = Path(n).parent.name or title
    elif p.is_dir():
        for f in sorted(p.rglob("*.txt")):
            items.append((str(f), f.read_text(encoding="utf-8", errors="replace")))
        title = p.name
    elif p.is_file():
        items.append((str(p), p.read_text(encoding="utf-8", errors="replace")))
    else:
        raise FileNotFoundError(path)
    eps = [parse_episode(t, n) for n, t in items]
    eps = [e for e in eps if e.events]
    eps.sort(key=lambda e: e.num)
    if not eps:
        raise ValueError("No parsable episodes found (expected '**LABEL:** / Start: (MM:SS.ss)' blocks).")
    return title, eps


def transcript(ep: Episode, rel_to: float | None = None, lo: float = 0.0, hi: float = 1e9) -> str:
    """Compact text for LLMs: dialogue + action only (camera notes dropped to save tokens)."""
    lines, scene_i = [], 0
    for ev in ep.content_events:
        if ev.end <= lo or ev.start >= hi:
            continue
        while scene_i < len(ep.scenes) and ep.scenes[scene_i].start <= ev.start + 1e-6 and rel_to is None:
            sc = ep.scenes[scene_i]
            lines.append(f"## SCENE {sc.n}: {sc.heading}")
            scene_i += 1
        s, e = (ev.start, ev.end) if rel_to is None else (max(ev.start, lo) - rel_to, min(ev.end, hi) - rel_to)
        stamp = f"[{fmt(s)}-{fmt(e)}]"
        if ev.kind == "dialogue":
            tone = f" ({ev.tone})" if ev.tone else ""
            lines.append(f'{stamp} {ev.speaker}{tone}: "{ev.text}"')
        elif ev.kind == "sfx":
            lines.append(f"{stamp} SFX: {ev.text}")
        else:
            lines.append(f"{stamp} ACTION: {ev.text}")
    return "\n".join(lines)
