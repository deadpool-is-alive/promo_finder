# Promo Finder (Kuku AI Generalist, Problem 1)

Reads microdrama screenplays and returns ranked 20-60s promo clips: **hook in the first 3s, cliffhanger at the cut,
understandable by someone who has never seen the show.** One command, any model.

## Run it (3 steps)
```bash
pip install requests                    # (+ streamlit pandas only for the optional UI)
cp .env.example .env                    # add your API key + model names (free Gemini/Groq tiers work)
python run.py --input scripts.zip       # zip, folder or single .txt; outputs land in ./outputs
```
Handy flags: `--episodes 1-5` (cheap dry run) · `--mode rules` (no API at all, free baseline) · `--max-stitched 0` (skip cross-episode)
· optional UI: `streamlit run app.py` · tests: `python tests/test_pipeline.py`.
No key set? It automatically runs the rules-only baseline.

## How it works
| Step | Who does it | What |
|---|---|---|
| 1 Parse | code | screenplay -> episodes/scenes/dialogue/action/camera with timestamps; camera notes dropped from LLM input (measured: 529k -> 196k characters, -63%) but kept as **clean cut points** |
| 2 Find | fast LLM (+ rules) | per episode: 2-4 promo proposals + short "beats". Rules add a safety net so a bad reply never leaves an episode empty |
| 2b Stitch | strong LLM | one call joins beats into cross-episode promos (2-3 segments) |
| 3 Clip | code | every timestamp is checked against the real script, snapped to shot boundaries, never cuts inside a line of dialogue, forced into 20-60s (trim setup / extend), duplicates collapsed. Invented timestamps are discarded and logged |
| 4 Judge | strong LLM | independent scoring of the **clip text only** (no show context): hook, cliffhanger, standalone clarity, spoiler risk, awkward cut |
| 5 Rank | code | weighted score (hook .35, cliff .30, standalone .25, no-spoiler .10), penalties for flags, near-duplicate removal, mild same-episode diversity penalty |
| 6 Output | code | `promos.csv/json`, `report.md`, `review_sheet.csv` (for your manual review), `run_meta.json` (tokens, models) |

**Swap the model:** edit `LLM_MODEL_FAST` / `LLM_MODEL_STRONG` / `LLM_BASE_URL` / `LLM_PROVIDER` in `.env`. Nothing else changes.

## Cost / scale (ESTIMATE from a mock-server dry run on the 50-episode pack, chars/4; your real `run_meta.json` has true counts)
~75 calls and ~100k input tokens per 50-episode show (pass A ~58k, judge ~37k, stitch ~3k). Responses are cached on disk (`.cache/`), so re-runs and
crashes cost nothing. For hundreds of shows: run pass A on the cheapest model, keep only the judge on the strong one, and have an editor
approve/reject (those decisions become few-shot examples in `prompts.py`).

## Known limits (be honest about these in your doc)
- The LLM path was verified with scripted/mock models and a local mock HTTP server, not a live API in the build sandbox. **Run it once with your key and read the output before submitting.**
- Scores come from an LLM reading text. It cannot see acting, music or visuals; it only sees what the script says.
- Hook/cliffhanger quality is judged by an LLM judge that is stricter than, but not the same as, a human editor.
- Cross-episode stitching is the weakest step; check those promos by hand.
- Rules-only baseline likes keywords and can end on a resolved moment (e.g. a wedding exit) and call it a cliffhanger.

## Confidentiality
The screenplay pack is Kuku's content. Do not commit the zip or the generated outputs to a public repo.
