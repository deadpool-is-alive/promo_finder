"""Optional UI:  streamlit run app.py   (upload the zip, press Run, download the ranked promos)."""
import os, tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from run import main as run_main

st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Promo Finder")
st.caption("Upload microdrama screenplays (.zip or .txt). Finds 20-60s promo clips: hook first, cliffhanger last.")

with st.sidebar:
    st.header("Model (swappable)")
    mode = st.radio("Mode", ["rules (no API)", "llm"], index=0 if not os.environ.get("LLM_API_KEY") else 1)
    provider = st.selectbox("Provider", ["openai", "anthropic"], index=0 if os.environ.get("LLM_PROVIDER", "openai") == "openai" else 1)
    base = st.text_input("Base URL", os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    key = st.text_input("API key", os.environ.get("LLM_API_KEY", ""), type="password")
    fast = st.text_input("Fast model (episode scan)", os.environ.get("LLM_MODEL_FAST", ""))
    strong = st.text_input("Strong model (judge + stitching)", os.environ.get("LLM_MODEL_STRONG", ""))
    eps = st.text_input("Episodes (blank = all, e.g. 1-5)", "1-5")

up = st.file_uploader("Screenplays", type=["zip", "txt"])
if up and st.button("Run", type="primary"):
    tmp = Path(tempfile.mkdtemp()); src = tmp / up.name; src.write_bytes(up.getvalue()); out = tmp / "out"
    os.environ.update(LLM_PROVIDER=provider, LLM_BASE_URL=base, LLM_API_KEY=key, LLM_MODEL_FAST=fast, LLM_MODEL_STRONG=strong or fast)
    argv = ["--input", str(src), "--out", str(out), "--mode", "llm" if mode == "llm" else "rules"] + (["--episodes", eps] if eps else [])
    box = st.status("Running ...", expanded=True)
    try:
        rows = run_main(argv, progress=lambda m: box.write(m))
        box.update(label="Done", state="complete")
    except SystemExit as e:
        box.update(label=f"Stopped: {e}", state="error"); st.stop()
    df = pd.DataFrame([{"rank": r["rank"], "where": " + ".join(f'E{x["episode"]} {x["start"]}-{x["end"]}' for x in r["segments"]),
                        "len_s": r["duration_s"], "score": r["scores"]["final_0_100"], "hook": r["hook"], "cliffhanger": r["cliffhanger"],
                        "why": r["why_it_works"], "verdict": r["verdict"], "flags": ", ".join(r["flags"])} for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True)
    for r in rows[:10]:
        with st.expander(f'#{r["rank"]}  E{r["segments"][0]["episode"]}  {r["duration_s"]:.0f}s: {r["hook"][:80]}'):
            st.code(r["clip_text"]); st.write(r["why_ranked_here"])
    c1, c2, c3 = st.columns(3)
    c1.download_button("promos.csv", (out / "promos.csv").read_bytes(), "promos.csv")
    c2.download_button("promos.json", (out / "promos.json").read_bytes(), "promos.json")
    c3.download_button("report.md", (out / "report.md").read_bytes(), "report.md")
