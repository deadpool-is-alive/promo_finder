"""The whole flow: parse -> candidates -> snap/validate -> judge -> score -> de-duplicate -> rank."""
from __future__ import annotations
import difflib, re, traceback
from concurrent.futures import ThreadPoolExecutor
from .clips import Cand, Seg, build_indexes, normalise, iou, clip_text, seg_str, MIN_LEN, MAX_LEN
from .heuristic import heuristic_candidates
from .parser import fmt, to_sec, transcript
from . import prompts

WEIGHTS = dict(hook=0.35, cliff=0.30, standalone=0.25, spoiler=0.10)


def _num(v, lo=0, hi=10, default=None):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- pass A
def episode_pass(llm, show, eps, ep, prev, log):
    user = (f"SHOW: {show}\nEPISODE {ep.num} of {len(eps)} | duration {fmt(ep.duration)}\n"
            f"Logline: {ep.logline}\nContinuity: {ep.continuity}\n"
            + (f"Previous episode logline: {prev.logline}\n" if prev else "")
            + f"\nTRANSCRIPT:\n{transcript(ep)}")
    try:
        data = llm.call(prompts.SYSTEM_A, user, "fast", 4000, "A_episode_candidates")
    except Exception as ex:
        log.append(f"[pass A] episode {ep.num}: LLM failed ({ex}); heuristic fallback only")
        return [], []
    promos, beats = [], []
    for n, p in enumerate(data.get("promos", [])[:4], 1):
        try:
            segs = [Seg(ep.num, to_sec(s["start"]), to_sec(s["end"])) for s in p["segments"]][:2]
            if not segs:
                raise ValueError("no segments")
            promos.append(Cand(f"L{ep.num}.{n}", segs, "llm", p.get("hook", ""), p.get("hook_quote", "") or "",
                               p.get("cliffhanger", ""), p.get("why", ""), _num(p.get("hook_score")), _num(p.get("cliff_score"))))
        except Exception as ex:
            log.append(f"[pass A] episode {ep.num}: dropped malformed promo #{n} ({ex})")
    for n, b in enumerate(data.get("beats", [])[:3], 1):
        try:
            s, e = to_sec(b["start"]), to_sec(b["end"])
            if 3 <= e - s <= 25 and b.get("role") in ("hook", "cliffhanger"):
                beats.append({"id": f"E{ep.num}.b{n}", "ep": ep.num, "start": s, "end": e,
                              "role": b["role"], "summary": str(b.get("summary", ""))[:200]})
        except Exception:
            pass
    return promos, beats


# ---------------------------------------------------------------- pass B
def stitch_pass(llm, eps, beats, idx, max_n, log):
    if len(beats) < 2 or max_n <= 0:
        return []
    for b in beats:                    # snap beats first so lengths are real
        ix = idx[b["ep"]]
        b["start"], b["end"] = ix.snap_start(b["start"]), ix.snap_end(b["end"])
    loglines = "\n".join(f"E{e.num}: {e.logline}" for e in eps)
    blist = "\n".join(f'{b["id"]} | {b["end"] - b["start"]:.0f}s | {b["role"]} | {b["summary"]}' for b in beats)
    user = f"EPISODES:\n{loglines}\n\nBEATS (id | length | role | summary):\n{blist}\n\nReturn at most {max_n} stitched promos."
    try:
        data = llm.call(prompts.SYSTEM_B, user, "strong", 3000, "B_stitch")
    except Exception as ex:
        log.append(f"[pass B] stitching failed ({ex})")
        return []
    by_id = {b["id"]: b for b in beats}
    out = []
    for n, s in enumerate(data.get("stitches", [])[:max_n], 1):
        ids = s.get("beat_ids", [])
        if not (2 <= len(ids) <= 3) or any(i not in by_id for i in ids) or len(set(ids)) != len(ids):
            log.append(f"[pass B] dropped stitch #{n}: bad/unknown beat ids {ids}")
            continue
        segs = [Seg(by_id[i]["ep"], by_id[i]["start"], by_id[i]["end"]) for i in ids]
        out.append(Cand(f"S{n}", segs, "stitched", s.get("hook", ""), "", s.get("cliffhanger", ""), s.get("why", "")))
    return out


# ---------------------------------------------------------------- pass C
def judge_pass(llm, cands, ep_by_num, batch, workers, log):
    groups = [cands[i:i + batch] for i in range(0, len(cands), batch)]

    def run(group):
        body = "\n\n".join(f'### CLIP id={c.cid} | total {c.dur:.0f}s\n{clip_text(c, ep_by_num)}' for c in group)
        try:
            data = llm.call(prompts.SYSTEM_C, body, "strong", 3500, "C_judge")
            return {str(r.get("id")): r for r in data.get("results", [])}
        except Exception as ex:
            log.append(f"[pass C] judge batch failed ({ex}); {len(group)} clips left unjudged")
            return {}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for group, res in zip(groups, ex.map(run, groups)):
            for c in group:
                r = res.get(c.cid)
                if r:
                    c.judge = {"hook": _num(r.get("hook_score")), "cliff": _num(r.get("cliff_score")),
                               "standalone": _num(r.get("standalone_score")), "spoiler": _num(r.get("spoiler_risk"), default=0),
                               "awkward_cut": bool(r.get("awkward_cut")), "verdict": r.get("verdict", "maybe"),
                               "hook_text": r.get("hook", ""), "cliff_text": r.get("cliffhanger", ""),
                               "why": r.get("why", ""), "confusion": r.get("confusion", "")}
                    if None in (c.judge["hook"], c.judge["cliff"], c.judge["standalone"]):
                        c.judge = {}


# ---------------------------------------------------------------- scoring / QA
def _norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def hook_quote_ok(c, ep_by_num):
    q = _norm(c.hook_quote)
    if not q:
        return None
    hay = _norm(clip_text(c, ep_by_num))
    if q in hay:
        return True
    lines = [_norm(re.sub(r"^\[.*?\]\s*", "", l)) for l in clip_text(c, ep_by_num).splitlines()]
    return any(difflib.SequenceMatcher(None, q, l).ratio() >= 0.8 or q in l for l in lines)


def opens_with_content(c, ep_by_num):
    s = c.segs[0]
    return any(e.start - s.start <= 3.0 and e.end > s.start for e in ep_by_num[s.ep].speech_events)


def final_score(c):
    j = c.judge
    if j:
        h, k, st, sp = j["hook"], j["cliff"], j["standalone"], j["spoiler"]
    else:
        h, k = c.prior_hook or 0, c.prior_cliff or 0
        st, sp = 5.0, 0.0
        c.flags.append("unjudged")
    s = 10 * (WEIGHTS["hook"] * h + WEIGHTS["cliff"] * k + WEIGHTS["standalone"] * st + WEIGHTS["spoiler"] * (10 - sp))
    if j and j.get("awkward_cut"): s *= 0.93; c.flags.append("awkward_cut")
    if "trimmed_to_fit" in c.flags or "extended_to_fit" in c.flags: s *= 0.97
    if j and j.get("verdict") == "skip": s *= 0.75; c.flags.append("judge_says_skip")
    if "hook_quote_unverified" in c.flags: s *= 0.95
    if "silent_open" in c.flags: s *= 0.9
    if j and j["standalone"] <= 4: c.flags.append("needs_context")
    return round(s, 1)


def rank(cands):
    remaining = sorted(cands, key=lambda c: -c.score)
    chosen, dropped = [], []
    while remaining:
        best = max(remaining, key=lambda c: c.score * (0.92 ** sum(1 for x in chosen if set(x.episodes) & set(c.episodes))))
        remaining.remove(best)
        if any(iou(best, x) >= 0.3 for x in chosen):
            best.flags.append("duplicate_of_higher_ranked")
            dropped.append(best)
            continue
        chosen.append(best)
        best.rank = len(chosen)
    return chosen, dropped


def why_ranked(c):
    j = c.judge
    if not j:
        return "Rule-based score only (no judge); lower confidence."
    bits = [f"hook {j['hook']:.0f}/10", f"cliffhanger {j['cliff']:.0f}/10", f"clear to newcomers {j['standalone']:.0f}/10"]
    extra = []
    if j["spoiler"] >= 6: extra.append("spoils a big beat")
    if j["spoiler"] <= 2: extra.append("no spoilers")
    if c.episodes[0] <= 5: extra.append("early episode = easy entry point for new viewers")
    if len(c.segs) > 1: extra.append("stitched")
    return ", ".join(bits) + ("; " + ", ".join(extra) if extra else "")


# ---------------------------------------------------------------- main
def run_pipeline(show, eps, llm=None, *, per_ep_heuristic=2, workers=4, judge_batch=4, max_stitched=8,
                 max_judged=150, log=None, progress=print):
    log = log if log is not None else []
    idx, ep_by_num = build_indexes(eps), {e.num: e for e in eps}
    cands, beats = [], []

    if llm is not None:
        progress(f"Pass A: reading {len(eps)} episodes with fast model ...")
        prevs = [None] + eps[:-1]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for promos, bts in ex.map(lambda a: episode_pass(llm, show, eps, a[0], a[1], log), zip(eps, prevs)):
                cands += promos; beats += bts
    for e in eps:                              # rules add a safety net / baseline
        cands += heuristic_candidates(e, idx[e.num], k=per_ep_heuristic if llm else 3)
    if llm is not None and max_stitched:
        progress(f"Pass B: stitching {len(beats)} beats into cross-episode promos ...")
        cands += stitch_pass(llm, eps, beats, idx, max_stitched, log)

    valid = []
    for c in cands:                            # Step 3: snap + validate (pure code)
        ok, why = normalise(c, idx)
        if ok:
            valid.append(c)
        else:
            log.append(f"[validate] dropped {c.cid} ({c.source}): {why}")
    # collapse near-identical candidates BEFORE paying for the judge
    valid.sort(key=lambda c: -((c.prior_hook or 0) + (c.prior_cliff or 0)))
    uniq = []
    for c in valid:
        if not any(iou(c, u) >= 0.7 for u in uniq):
            uniq.append(c)
    uniq = uniq[:max_judged]
    progress(f"{len(cands)} raw candidates -> {len(valid)} valid -> {len(uniq)} distinct")

    for c in uniq:
        okq = hook_quote_ok(c, ep_by_num)
        if okq is False: c.flags.append("hook_quote_unverified")
        if not opens_with_content(c, ep_by_num): c.flags.append("silent_open")

    if llm is not None:
        progress(f"Pass C: independent judge on {len(uniq)} clips (strong model) ...")
        judge_pass(llm, uniq, ep_by_num, judge_batch, workers, log)
    for c in uniq:
        c.score = final_score(c)
    chosen, dropped = rank(uniq)
    return chosen, dropped, ep_by_num, log
