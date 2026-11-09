"""Step 6: write the outputs an editor (and a reviewer) can actually use."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path
from .clips import build_indexes, clip_text, seg_str
from .parser import fmt
from .pipeline import hook_quote_ok, opens_with_content, why_ranked


def promo_dict(c, ep_by_num, idx):
    j = c.judge
    kind = "single" if len(c.segs) == 1 else ("multi_same_episode" if len(set(s.ep for s in c.segs)) == 1 else "cross_episode")
    qa = {
        "duration_20_60": 20 - 0.01 <= c.dur <= 60 + 0.01,
        "within_episode": all(0 <= s.start < s.end <= ep_by_num[s.ep].duration + 0.01 for s in c.segs),
        "clean_cuts_no_mid_dialogue": all(not idx[s.ep].inside_dialogue(s.start) and not idx[s.ep].inside_dialogue(s.end) for s in c.segs),
        "opens_with_content_in_3s": opens_with_content(c, ep_by_num),
        "hook_quote_found_in_clip": hook_quote_ok(c, ep_by_num),
    }
    return {
        "rank": c.rank, "promo_id": c.cid, "type": kind, "source": c.source, "duration_s": round(c.dur, 1),
        "segments": [{"episode": s.ep, "start": fmt(s.start), "end": fmt(s.end),
                      "start_s": round(s.start, 2), "end_s": round(s.end, 2)} for s in c.segs],
        "hook": j.get("hook_text") or c.hook, "cliffhanger": j.get("cliff_text") or c.cliff,
        "why_it_works": j.get("why") or c.why, "why_ranked_here": why_ranked(c),
        "scores": {"final_0_100": c.score, "hook": j.get("hook", c.prior_hook), "cliffhanger": j.get("cliff", c.prior_cliff),
                   "standalone": j.get("standalone"), "spoiler_risk": j.get("spoiler"), "judged": bool(j)},
        "verdict": j.get("verdict", "unjudged"), "newcomer_confusion": j.get("confusion", ""),
        "flags": sorted(set(c.flags)), "qa_checks": qa, "clip_text": clip_text(c, ep_by_num),
    }


def write_outputs(out_dir, show, eps, chosen, dropped, ep_by_num, log, llm, meta):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    idx = build_indexes(eps)
    rows = [promo_dict(c, ep_by_num, idx) for c in chosen]
    (out / "promos.json").write_text(json.dumps({"show": show, "meta": meta, "promos": rows}, indent=2, ensure_ascii=False))

    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "promo_id", "type", "episodes", "segments (episode start-end)", "duration_s", "hook", "cliffhanger",
                    "why_it_works", "why_ranked_here", "final_score", "hook_score", "cliff_score", "standalone_score",
                    "spoiler_risk", "verdict", "flags"])
        for r in rows:
            s = r["scores"]
            w.writerow([r["rank"], r["promo_id"], r["type"], ",".join(str(x["episode"]) for x in r["segments"]),
                        " | ".join(f'E{x["episode"]} {x["start"]}-{x["end"]}' for x in r["segments"]), r["duration_s"],
                        r["hook"], r["cliffhanger"], r["why_it_works"], r["why_ranked_here"], s["final_0_100"], s["hook"],
                        s["cliffhanger"], s["standalone"], s["spoiler_risk"], r["verdict"], ";".join(r["flags"])])

    with open(out / "review_sheet.csv", "w", newline="", encoding="utf-8") as f:   # for the manual quality check
        w = csv.writer(f)
        w.writerow(["rank", "promo_id", "segments", "clip_text", "HUMAN: hook (good/ok/bad)", "HUMAN: cliffhanger (good/ok/bad)",
                    "HUMAN: standalone (good/ok/bad)", "HUMAN: would you post it? (y/n)", "HUMAN: notes / failure reason"])
        for r in rows[:30]:
            w.writerow([r["rank"], r["promo_id"], " | ".join(f'E{x["episode"]} {x["start"]}-{x["end"]}' for x in r["segments"]),
                        r["clip_text"], "", "", "", "", ""])

    # ---- automated "where does it pick badly" stats
    n = len(rows) or 1
    def cnt(flag): return sum(1 for r in rows if flag in r["flags"])
    qa_fail = {k: sum(1 for r in rows if r["qa_checks"][k] is False) for k in rows[0]["qa_checks"]} if rows else {}
    eps_covered = {e for r in rows[:40] for e in (x["episode"] for x in r["segments"])}
    top_ep_counts = {}
    for r in rows[:15]:
        for x in {x["episode"] for x in r["segments"]}:
            top_ep_counts[x] = top_ep_counts.get(x, 0) + 1
    scores = [r["scores"]["final_0_100"] for r in rows]
    judged = sum(1 for r in rows if r["scores"]["judged"])

    L = [f"# Promo report: {show}", "",
         f"Run: {meta['mode']} | episodes: {len(eps)} | promos ranked: {len(rows)} | judged by LLM: {judged}/{len(rows)}",
         f"Models: fast=`{meta.get('fast_model','-')}` strong=`{meta.get('strong_model','-')}` | tokens in/out: {meta['usage']['input_tokens']}/{meta['usage']['output_tokens']} | LLM calls: {meta['usage']['calls']} (cache hits {meta['usage']['cache_hits']})", "",
         "## Post these first (top 10)", "",
         "| # | Where | Len | Hook | Cliffhanger | Why it works | Why ranked here |", "|---|---|---|---|---|---|---|"]
    clean = lambda t: str(t).replace("|", "/").replace("\n", " ")[:170]
    for r in rows[:10]:
        where = "<br>".join(f'E{x["episode"]} {x["start"]}-{x["end"]}' for x in r["segments"])
        L.append(f'| {r["rank"]} | {where} | {r["duration_s"]:.0f}s | {clean(r["hook"])} | {clean(r["cliffhanger"])} | {clean(r["why_it_works"])} | {clean(r["why_ranked_here"])} |')
    L += ["", "## Full ranking", "", "| # | Where | Len | Score | Type | Verdict | Flags |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        where = " + ".join(f'E{x["episode"]} {x["start"]}-{x["end"]}' for x in r["segments"])
        L.append(f'| {r["rank"]} | {where} | {r["duration_s"]:.0f}s | {r["scores"]["final_0_100"]} | {r["type"]} | {r["verdict"]} | {", ".join(r["flags"])} |')
    L += ["", "## Automatic quality checks (where the tool is shaky)", "",
          f"- Hard-rule checks failing: {json.dumps(qa_fail)}",
          f"- Needs earlier context (judge standalone <= 4): {cnt('needs_context')} of {len(rows)}",
          f"- Awkward cut flagged by judge: {cnt('awkward_cut')} | silent open (nothing in first 3s): {cnt('silent_open')}",
          f"- Hook line the model quoted was NOT found in the clip (possible hallucination): {cnt('hook_quote_unverified')}",
          f"- Duration auto-repaired (trimmed/extended): {cnt('trimmed_to_fit') + cnt('extended_to_fit')}",
          f"- Not judged by LLM (fallback scores only): {cnt('unjudged')}",
          f"- Episodes represented in top 40: {len(eps_covered)}/{len(eps)}; most-used episodes in top 15: {sorted(top_ep_counts.items(), key=lambda kv: -kv[1])[:3]}",
          f"- Score spread: min {min(scores, default=0)}, median {statistics.median(scores) if scores else 0}, max {max(scores, default=0)} (a flat spread means the judge is not discriminating)",
          "", "## Run log (dropped/failed items)", ""]
    L += [f"- {x}" for x in log[:80]] or ["- none"]
    L += ["", "## Human review", "", "Fill in `review_sheet.csv` for the top 10 and bottom picks, then record what you found in the solution doc."]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "run_log.txt").write_text("\n".join(log), encoding="utf-8")
    return rows
