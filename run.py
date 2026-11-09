#!/usr/bin/env python3
"""One command: screenplays in -> ranked promos out.

  python run.py --input scripts.zip                      # uses LLM if LLM_API_KEY is set, else rules only
  python run.py --input scripts.zip --episodes 1-5       # cheap dry run on 5 episodes
  python run.py --input scripts.zip --mode rules         # no API, free baseline
"""
import argparse, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pf.parser import load_episodes
from pf.llm import LLM
from pf.pipeline import run_pipeline
from pf.report import write_outputs


def load_dotenv(p=".env"):
    if Path(p).exists():
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_range(spec, nums):
    if not spec:
        return set(nums)
    keep = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        keep |= set(range(int(a), int(b or a) + 1))
    return keep & set(nums)


def main(argv=None, progress=print):
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="zip, folder or single .txt of screenplays")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--mode", choices=["auto", "llm", "rules"], default="auto")
    ap.add_argument("--episodes", default="", help="e.g. 1-5,9 (default: all)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LLM_WORKERS", "4")))
    ap.add_argument("--judge-batch", type=int, default=4)
    ap.add_argument("--max-stitched", type=int, default=8, help="cross-episode promos to attempt (0 = off)")
    ap.add_argument("--heuristic-per-episode", type=int, default=2)
    a = ap.parse_args(argv)

    show, eps = load_episodes(a.input)
    keep = parse_range(a.episodes, [e.num for e in eps])
    eps = [e for e in eps if e.num in keep]
    progress(f"Loaded '{show}': {len(eps)} episodes, {sum(len(e.events) for e in eps)} timed events")

    use_llm = a.mode == "llm" or (a.mode == "auto" and os.environ.get("LLM_API_KEY"))
    llm = LLM.from_env() if use_llm else None
    if a.mode == "llm" and not os.environ.get("LLM_API_KEY"):
        sys.exit("LLM_API_KEY not set. See .env.example, or use --mode rules.")
    if llm and not llm.models["fast"]:
        sys.exit("LLM_MODEL_FAST not set. See .env.example.")
    if llm:
        for tier in ("fast", "strong"):
            try:
                llm.call("Reply with JSON only.", 'Return {"ok": true}', tier, 1000, "preflight")
            except Exception as ex:
                sys.exit(f"Model check failed for {tier} model '{llm.models[tier]}': {ex}")
        # progress("No LLM configured: running rules-only baseline (set LLM_API_KEY for the full pipeline).")

    t0, log = time.time(), []
    chosen, dropped, ep_by_num, log = run_pipeline(show, eps, llm, per_ep_heuristic=a.heuristic_per_episode, workers=a.workers,
                                                   judge_batch=a.judge_batch, max_stitched=a.max_stitched, log=log, progress=progress)
    meta = {"mode": "llm+rules" if llm else "rules-only", "fast_model": llm.models["fast"] if llm else None,
            "strong_model": llm.models["strong"] if llm else None, "provider": llm.provider if llm else None,
            "seconds": round(time.time() - t0, 1), "episodes_run": [e.num for e in eps],
            "usage": llm.usage if llm else {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0},
            "tokens_by_stage": llm.by_label if llm else {}}
    rows = write_outputs(a.out, show, eps, chosen, dropped, ep_by_num, log, llm, meta)
    progress(f"Done in {meta['seconds']}s -> {a.out}/ (report.md, promos.csv, promos.json, review_sheet.csv)")
    (Path(a.out) / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return rows


if __name__ == "__main__":
    main()
