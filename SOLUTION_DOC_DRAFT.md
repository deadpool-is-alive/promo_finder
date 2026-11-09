# Promo Finder: solution doc (draft, trim to 2 pages, export to PDF)
**Problem 1.** <name, roll no, branch> · Prototype: <link> · Video: <link>

## Why this problem, how I scoped it
An editor today watches every episode by hand, and new shows land weekly. The expensive part is *finding* the 20-60s moments, so I built only that: right moments, right start/end. No editing, music or overlays.
Show used: *I Married a Broken Soldier. He was a Duke* (50 episodes, avg ~2 min each).

## Assumptions (brief was silent)
- Timestamps are per-episode and continuous across scenes (verified in the pack). Cuts snap to camera-shot boundaries and never split a line of dialogue.
- The screenplay is a faithful proxy for the video; I cannot see acting, music or visuals.
- "Makes sense alone" = judged on the clip text only, with no show summary given to the judge.
- Ranking weights (hook .35 / cliffhanger .30 / standalone .25 / no-spoiler .10) are my judgement, set to match the brief's three criteria; an editor should tune them.
- Cross-episode promos use short "beats" (4-18s) so 2-3 can fit under 60s.

## How it works (code does mechanics, AI does judgement)
1. **Parse (code):** screenplay -> timed events. Camera notes are removed from LLM input (-63% characters) but kept as clean cut points.
2. **Find (fast LLM + rules):** one call per episode proposes 2-4 promos + beats; rules add a fallback so no episode is empty.
3. **Stitch (strong LLM):** one call joins beats across episodes.
4. **Clip (code):** every timestamp checked against the real script; snapped; forced into 20-60s; invented timestamps rejected and logged.
5. **Judge (strong LLM, clip-only view):** hook, cliffhanger, standalone, spoiler risk, awkward cut, one-line why.
6. **Rank (code):** weighted score, flag penalties, duplicate removal, mild same-episode diversity penalty.
Model-swappable: change env vars only (OpenAI-compatible or Anthropic API).

## Scale and cost
~75 calls, ~<FILL from run_meta.json> tokens per 50-episode show; cached on disk so reruns are free. At many shows: cheap model for step 2, strong model only on the ~100 shortlisted clips, editor approves/rejects, and approved clips become few-shot examples in the prompts. Cost scales linearly with episodes (judge cost with candidates, which I cap).

## How I know it is good (and where it is not)
- **Hard checks (automatic, in report.md):** duration 20-60s, inside episode, no mid-dialogue cuts, opens with speech/action in 3s, quoted hook line really exists in the clip. <FILL: pass rates>
- **Human review:** I scored the top 10 + 10 random lower picks as good/ok/bad on hook, cliffhanger, standalone (`review_sheet.csv`). <FILL: e.g. "7/10 top picks I would post">
- **Baseline comparison:** rules-only vs LLM pipeline on the same 50 episodes. <FILL: overlap of top 10, what the LLM fixed>
- **Known failures:** <FILL from your review>. Seen already in the rules baseline: it ranked a resolved wedding-exit (E4) as a cliffhanger because the line before ended in "..."; it rewards keyword-heavy lines over real stakes. Stitched promos are the weakest (<FILL>).

## AI tools used + one place they got it wrong
Claude (design, code, prompts), <others>. **Wrong:** the first parser treated `SOUND FX` as a character's dialogue, so a "carriage door slamming" cue became a ranked hook; I only caught it by reading the baseline's top 10, then added a separate `sfx` event type. <replace with your own example if you have one>
