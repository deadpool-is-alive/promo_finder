"""Run: python -m pytest -q   (or python tests/test_pipeline.py). Needs the screenplay zip path in PF_TEST_INPUT,
otherwise uses a tiny built-in sample so the repo's tests never need confidential data."""
import json, os, re, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pf.parser import parse_episode, load_episodes, to_sec, fmt
from pf.clips import build_indexes, Cand, Seg, normalise
from pf.llm import LLM, extract_json
from pf.pipeline import run_pipeline

SAMPLE = """EPISODE NUMBER: Episode 1
EPISODE LOGLINE: A bride is dumped at the altar.
CONTINUITY NOTES: First episode.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CATHEDRAL – DAY
Scene start: 00:00.00
Scene end: 00:40.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:04.00)
Wide shot of the doors.

**MATRIARCH (Cold):**
Start: (00:01.00) | End: (00:03.50)
"You should turn around."

**CAMERA:**
Start: (00:04.00) | End: (00:12.00)
Close-up on the bride.

**LADY JULIANA (Shaken):**
Start: (00:04.50) | End: (00:11.00)
"Gerald is waiting for me, and he would never betray me, would he?"

**SOUND FX:**
Start: (00:12.00) | End: (00:14.00)
"A heavy sting."

**CAMERA:**
Start: (00:12.00) | End: (00:40.00)
Long hold on the altar.

**GERALD (Smug):**
Start: (00:15.00) | End: (00:20.00)
"Because I am marrying her instead!"

**ACTION:**
Start: (00:21.00) | End: (00:39.00)
Juliana slaps Gerald across the face and the guests gasp.

--- SCRIPT END ---
"""


def sample_eps():
    zp = os.environ.get("PF_TEST_INPUT")
    if zp:
        return load_episodes(zp)[1]
    return [parse_episode(SAMPLE.replace("Episode 1", f"Episode {n}"), f"ep{n}") for n in (1, 2, 3)]


def test_parser():
    e = parse_episode(SAMPLE)
    kinds = [x.kind for x in e.events]
    assert kinds.count("dialogue") == 3 and kinds.count("sfx") == 1 and kinds.count("action") == 1
    assert e.duration == 40.0 and e.num == 1 and len(e.scenes) == 1
    assert to_sec("01:02.50") == 62.5 and fmt(62.5) == "01:02.50" and fmt(59.999) == "01:00.00"


def test_snapping_never_cuts_dialogue_and_forces_20_60():
    eps = sample_eps(); idx = build_indexes(eps)
    for ep in eps[:10]:
        ix = idx[ep.num]
        for s, e in [(5.2, 9.0), (0.3, 70), (7.7, 200)]:
            c = Cand("t", [Seg(ep.num, min(s, ep.duration - 2), min(e, ep.duration))], "llm")
            ok, _ = normalise(c, idx)
            if ok:
                assert 19.99 <= c.dur <= 60.01
                for sg in c.segs:
                    assert not ix.inside_dialogue(sg.start) and not ix.inside_dialogue(sg.end)


class FakeLLM:
    """Scripted model. Deliberately includes garbage a real model might return."""
    provider, models = "fake", {"fast": "fake-fast", "strong": "fake-strong"}
    def __init__(self): self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}; self.by_label = {}
    def call(self, system, user, tier="fast", max_tokens=0, label=""):
        self.usage["calls"] += 1
        if label == "A_episode_candidates":
            ts = re.findall(r"\[(\d\d:\d\d\.\d\d)-(\d\d:\d\d\.\d\d)\]", user)
            first, last = ts[0][0], ts[-1][1]
            return {"promos": [
                {"segments": [{"start": first, "end": last}], "hook": "slap", "hook_quote": "Because I am marrying her instead!",
                 "cliffhanger": "guests gasp", "why": "w", "hook_score": 9, "cliff_score": 8},
                {"segments": [{"start": "99:00.00", "end": "99:30.00"}], "hook": "invented", "hook_score": 10},   # out of range
                {"segments": [], "hook": "malformed"},
                {"segments": [{"start": "oops", "end": "??"}]}],
                "beats": [{"start": ts[0][0], "end": ts[1][1], "role": "hook", "summary": "s"},
                          {"start": ts[-2][0], "end": ts[-1][1], "role": "cliffhanger", "summary": "c"}]}
        if label == "B_stitch":
            ids = re.findall(r"^(E\d+\.b\d+) ", user, re.M)
            return {"stitches": [{"beat_ids": ids[:2], "hook": "h", "cliffhanger": "c", "why": "w"},
                                 {"beat_ids": ["E99.b1", "E1.b1"]}]}                                  # unknown id
        if label == "C_judge":
            ids = re.findall(r"### CLIP id=(\S+)", user)
            return {"results": [{"id": i, "hook_score": 7, "cliff_score": 6, "standalone_score": 8, "spoiler_risk": 1,
                                 "awkward_cut": False, "hook": "h", "cliffhanger": "c", "why": "w", "confusion": "", "verdict": "post"}
                                for i in ids[:-1]]}                                                    # last one missing -> unjudged
        raise AssertionError(label)


def test_full_pipeline_with_fake_llm_survives_garbage():
    eps = sample_eps()
    log = []
    chosen, dropped, by_num, log = run_pipeline("Test", eps, FakeLLM(), log=log, progress=lambda *_: None, judge_batch=3, workers=2)
    assert chosen and [c.rank for c in chosen] == list(range(1, len(chosen) + 1))
    assert all(19.99 <= c.dur <= 60.01 for c in chosen)
    assert any("unjudged" in c.flags for c in chosen)          # missing judge result is handled, not fatal
    assert any("dropped" in l for l in log)                    # garbage was rejected and logged, not crashed on


def test_rules_only_pipeline():
    chosen, *_ = run_pipeline("Test", sample_eps(), None, progress=lambda *_: None)
    assert isinstance(chosen, list)


def test_extract_json_variants():
    assert extract_json('```json\n{"a":1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": [1,2]} thanks') == {"a": [1, 2]}


def _server(handler_factory):
    srv = HTTPServer(("127.0.0.1", 0), handler_factory); threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv


def test_llm_client_openai_and_anthropic_formats_retry_and_cache(tmp_path):
    state = {"hits": 0}
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            state["hits"] += 1
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if state["hits"] == 1:                              # first call: rate limited
                self.send_response(429); self.send_header("retry-after", "0"); self.end_headers(); self.wfile.write(b"{}"); return
            if self.path.endswith("/chat/completions"):
                out = {"choices": [{"message": {"content": '```json\n{"ok": "openai"}\n```'}}], "usage": {"prompt_tokens": 11, "completion_tokens": 5}}
            else:
                assert self.headers["x-api-key"] == "k" and "system" in body
                out = {"content": [{"type": "text", "text": '{"ok": "anthropic"}'}], "usage": {"input_tokens": 7, "output_tokens": 3}}
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(json.dumps(out).encode())
    srv = _server(H); url = f"http://127.0.0.1:{srv.server_port}"
    a = LLM("openai", url, "k", "m", "m", str(tmp_path / "c1"))
    assert a.call("s", "u") == {"ok": "openai"} and a.usage["input_tokens"] == 11
    n = state["hits"]; assert a.call("s", "u") == {"ok": "openai"} and state["hits"] == n and a.usage["cache_hits"] == 1   # cached
    b = LLM("anthropic", url, "k", "m", "m", str(tmp_path / "c2"))
    assert b.call("s", "u") == {"ok": "anthropic"} and b.usage["output_tokens"] == 3
    srv.shutdown()


if __name__ == "__main__":
    import tempfile, traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(Path(tempfile.mkdtemp())) if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount] else fn()
                print("PASS", name)
            except Exception:
                fails += 1; print("FAIL", name); traceback.print_exc()
    sys.exit(1 if fails else 0)
