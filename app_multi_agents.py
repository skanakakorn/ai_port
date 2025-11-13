
import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============== Cheap LLM Adapter ===================
def call_llm_cheap(prompt, system=None, temperature=0.2, max_tokens=350):
    provider = os.getenv("LLM_PROVIDER", "DUMMY").upper()
    model = os.getenv("LLM_MODEL_ID", "dummy-mini")
    if provider == "OPENAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model=model, messages=msgs,
                temperature=temperature, max_completion_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM error OPENAI: {e}]"
    elif provider == "TOGETHER":
        try:
            import requests
            url = "https://api.together.xyz/v1/chat/completions"
            headers = {"Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}", "Content-Type": "application/json"}
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            payload = {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[LLM error TOGETHER: {e}]"
    elif provider == "GROQ":
        try:
            import requests
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}", "Content-Type": "application/json"}
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            payload = {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[LLM error GROQ: {e}]"
    else:
        return ("[DUMMY LLM]\n"
                "This is a placeholder response. Set LLM_PROVIDER and API key "
                "to get real answers (e.g., OPENAI/gpt-4o-mini, TOGETHER/llama-3.1-8b-instruct).")

# ============== Pre-baked Theme Rationales ==============
THEME_INFO = {
    "AI Compute": "Supplies GPUs/CPUs and adjacent silicon (e.g., NVIDIA, AMD). Drivers: model scaling, datacenter capex. Risks: supply chain, pricing cycles, inventory corrections.",
    "AI Foundry": "Manufactures chips at scale (TSMC). Moat: process leadership, yield. Risks: geopolitics, customer concentration.",
    "AI Equipment": "Lithography and tools (ASML). Demand tied to leading-edge nodes. Risks: export controls, order volatility.",
    "AI Networking": "High-speed interconnects and NICs (AVGO, MRVL). Bottleneck reduction for GPU clusters. Risks: hyperscaler bargaining power.",
    "AI Servers": "System integration (SMCI). Benefits from rapid GPU rack growth. Risks: margin compression as competition rises.",
    "AI Platform": "Hyperscale AI APIs and copilots (MSFT, GOOGL, AMZN). Strong distribution; risks: GPU cost pressure, cannibalization.",
    "Data Platform": "Storage/analytics fabric (SNOW, MDB). Growth from AI apps; risks: cost optimization by customers.",
    "Enterprise AI": "Workflow and automation (NOW, CRM, ADBE, SAP). Advantage: installed base + data. Risks: pricing pressure, AI-native challengers.",
    "MLOps/Observability": "Operations and monitoring (DDOG). AI increases telemetry; risks: budget scrutiny, competition.",
    "Security": "Zero trust / inspection for AI-era traffic (ZS). Risks: seat growth sensitivity, platform convergence.",
    "AI-Enabled Consumer": "Commerce/fintech using AI for recsys and risk (SE, SHOP). Risks: macro cycles, competition, take-rate pressure.",
    "Memory": "DRAM/NAND for AI servers (MU). Cyclical ASPs; benefits from HBM demand."
}

# ============== Streamlit App ==============
st.set_page_config(page_title="AI Portfolio Agent — Multi-Agent", page_icon="🧩")
st.title("🧩 AI Portfolio Agent — Multi-Agent Committee")
#st.caption("Analyst vs Contrarian with a Consensus summary. Cheap, fast, demo-ready.")

# ----- Data load -----
st.sidebar.header("1) Load Universe")

@st.cache_data
def load_default(path: str, mtime: float):
    return pd.read_csv(path)

default_path = "ai_universe_sample_final.csv"
st.sidebar.info("Using bundled sample: ai_universe_sample_final.csv")
try:
    mtime = os.path.getmtime(default_path)
except Exception:
    mtime = 0.0
universe = load_default(default_path, mtime)

# Basic normalization to avoid hidden whitespace/case issues
for col in ["ticker", "name", "theme", "mcap_bucket"]:
    if col in universe.columns:
        universe[col] = universe[col].astype(str).str.strip()

needed_cols = {"ticker","name","theme","mcap_bucket","expected_return","volatility","momentum_6m"}
if not needed_cols.issubset(set(universe.columns)):
    st.error(f"CSV must include columns: {sorted(list(needed_cols))}")
    st.stop()

# ----- Strategy controls -----
st.sidebar.header("2) Strategy Controls")
n_names = st.sidebar.slider("Stocks per portfolio", 6, 20, 10, 1)
max_weight = st.sidebar.slider("Max weight per name", 0.05, 0.30, 0.15, 0.01)
risk_floor = st.sidebar.slider("Volatility floor (optional)", 0.0, 0.6, 0.0, 0.01)
all_themes = sorted(universe["theme"].unique().tolist())
selected_themes = st.sidebar.multiselect("Filter themes (optional)", options=all_themes, default=[])

def apply_filters(df):
    x = df.copy()
    if selected_themes:
        x = x[x["theme"].isin(selected_themes)]
    if risk_floor > 0:
        x = x[x["volatility"] >= risk_floor]
    return x.reset_index(drop=True)

base = apply_filters(universe)

st.subheader("Universe (after filters)")
st.dataframe(base.sort_values(["theme","volatility"]).reset_index(drop=True), use_container_width=True, hide_index=True)

# ----- Portfolio constructors -----
def cap_weights(w, cap):
    w = np.array(w, dtype=float)
    if w.sum() == 0:
        return w
    w = w / w.sum()
    for _ in range(5):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if under.any():
            w[under] += (excess / under.sum())
        w = np.clip(w, 0, cap)
        if abs(w.sum() - 1.0) < 1e-9:
            break
        w = w / w.sum()
    return w

def p_low_vol(df, n):
    pick = df.nsmallest(n, "volatility").copy()
    inv = 1.0 / np.maximum(pick["volatility"].values, 1e-6)
    w = inv / inv.sum()
    w = cap_weights(w, max_weight)
    pick["weight"] = w
    pick["strategy"] = "A) Defensive Low-Vol Infra"
    return pick

def p_balanced(df, n):
    df = df.copy()
    df["score"] = df["expected_return"] / np.maximum(df["volatility"], 1e-6)
    pick = df.nlargest(n, "score").copy()
    w = pick["score"].values
    w = w / w.sum()
    w = cap_weights(w, max_weight)
    pick["weight"] = w
    pick["strategy"] = "B) Balanced Sharpe Mix"
    return pick.drop(columns=["score"])

def p_momentum(df, n):
    pick = df.nlargest(n, "momentum_6m").copy()
    m = pick["momentum_6m"].values - pick["momentum_6m"].min() + 1e-6
    w = m / m.sum()
    w = cap_weights(w, max_weight)
    pick["weight"] = w
    pick["strategy"] = "C) High-Conviction Momentum"
    return pick

if len(base) < n_names:
    st.warning("Not enough names after filters. Reduce 'Stocks per portfolio' or widen filters.")
    st.stop()

pA, pB, pC = p_low_vol(base, n_names), p_balanced(base, n_names), p_momentum(base, n_names)
result = pd.concat([pA, pB, pC], axis=0, ignore_index=True)

def summarize_portfolio(df):
    w = df["weight"].values
    er = (df["expected_return"].values * w).sum()
    vol = (df["volatility"].values * w).sum()
    theme_mix = df.groupby("theme")["weight"].sum().sort_values(ascending=False)
    return er, vol, theme_mix

st.subheader("Portfolios")
tabs = st.tabs([t for t,_ in result.groupby("strategy")])
for tab, (label, pf) in zip(tabs, result.groupby("strategy")):
    with tab:
        st.markdown(f"**{label}**")
        st.dataframe(
            pf[["ticker","name","theme","expected_return","volatility","momentum_6m","weight"]]
            .sort_values("weight", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True
        )
        exp_r, vol, theme_mix = summarize_portfolio(pf)
        st.write(f"Expected Return (heuristic): **{exp_r:.2%}**  |  Volatility (heuristic): **{vol:.2%}**")

        fig, ax = plt.subplots()
        ax.bar(pf["ticker"], pf["weight"])
        ax.set_title(f"Weights — {label}")
        ax.set_xlabel("Ticker"); ax.set_ylabel("Weight")
        st.pyplot(fig)

        st.markdown("**Theme Exposure**")
        fig2, ax2 = plt.subplots()
        ax2.bar(theme_mix.index, theme_mix.values)
        ax2.set_xticklabels(theme_mix.index, rotation=30, ha="right")
        ax2.set_ylabel("Weight")
        st.pyplot(fig2)

# ============== Multi-Agent Explainer ==============
st.subheader("🤝 Multi-Agent Committee for a Selected Stock")
picked = result[["ticker","name","theme","mcap_bucket","expected_return","volatility","momentum_6m","strategy","weight"]].copy()
tickers = picked["ticker"].unique().tolist()
if not tickers:
    st.info("No picks available yet.")
    st.stop()

col1, col2 = st.columns([1,2])
with col1:
    chosen = st.selectbox("Ticker", tickers)
    row = picked[picked["ticker"] == chosen].iloc[0].to_dict()
    st.markdown(f"**{row['ticker']} — {row['name']}**")
    st.markdown(
        f"- Theme: {row['theme']}  \n"
        f"- Market-cap: {row['mcap_bucket']}  \n"
        f"- Strategy: {row['strategy']}  \n"
        f"- Weight: {row['weight']:.1%}  \n"
        f"- Exp. Return: {row['expected_return']:.1%}  \n"
        f"- Volatility: {row['volatility']:.1%}  \n"
        f"- 6M Momentum: {row['momentum_6m']:.1%}"
    )
    st.markdown(f"**Theme rationale (pre-baked):** {THEME_INFO.get(row['theme'], '—')}")
with col2:
    st.markdown("**Question to the committee**")
    q = st.text_input("e.g., Why is this suitable for the chosen strategy, and what should I watch out for?",
                      value="Why is this suitable for the chosen strategy, and what are top 2 risks?")

analyst_sys = (
    "You are an equity ANALYST agent. Use ONLY the provided facts and short theme rationale. "
    "Explain succinctly why the stock fits the strategy. <=160 words."
)
contrarian_sys = (
    "You are a CONTRARIAN risk agent. Use ONLY provided facts and theme rationale. "
    "List only 2-3 plausible risks or counterpoints. <=150 words."
)

# Synthesis/consensus agent guidelines
consensus_sys = (
    "You are the COMMITTEE CHAIR synthesizing viewpoints. Provide a balanced, neutral "
    "summary that combines the analyst rationale with the contrarian risks in 2–3 "
    "sentences. Avoid hype and real-time claims. <=120 words."
)

compact = (
    f"Ticker:{row['ticker']} Name:{row['name']} Theme:{row['theme']} Cap:{row['mcap_bucket']} "
    f"Strategy:{row['strategy']} Weight:{row['weight']:.3f} "
    f"ER:{row['expected_return']:.3f} Vol:{row['volatility']:.3f} Mom6M:{row['momentum_6m']:.3f}\n"
    f"ThemeNotes:{THEME_INFO.get(row['theme'],'')}"
)
analyst_prompt = (
    "Context:\n" + compact +
    "\nTask: Answer the user's question briefly as an analyst in Thai language. Do not claim real-time info. "
    f"User question: {q}"
)

contrarian_prompt = (
    "Context:\n" + compact +
    "\nTask: Challenge the bullish view in Thai language with 2 concise counterpoints to the strategy."
    "No real-time claims. "
    f"User question: {q}"
)

xcontrarian_prompt = (
    "Context:\n" + compact +
    "\nTask: Challenge the bullish view with 2–3 concise risks relevant to the strategy. "
    "No real-time claims. "
    f"User question: {q}"
)

cache_key = f"{row['ticker']}::{q}::{os.getenv('LLM_PROVIDER','DUMMY')}::{os.getenv('LLM_MODEL_ID','dummy')}"
if "ma_cache" not in st.session_state:
    st.session_state.ma_cache = {}
if cache_key not in st.session_state.ma_cache:
    analyst = call_llm_cheap(analyst_prompt, system=analyst_sys, temperature=0.2, max_tokens=400)
    contrarian = call_llm_cheap(contrarian_prompt, system=contrarian_sys, temperature=0.2, max_tokens=400)
    consensus_prompt = (
        "Context:\n" + compact +
        "\nAnalyst view:\n" + analyst +
        "\nContrarian view:\n" + contrarian +
        "\nTask: Write a concise synthesis (4-5 sentences) in Thai language that captures the key "
        "rationale and top risks for this stock within the chosen strategy. Tell whether"
        "the stock is buy/sell/hold. No "
        "real-time claims."
    )
    consensus = call_llm_cheap(consensus_prompt, system=consensus_sys, temperature=0.2, max_tokens=900)
    st.session_state.ma_cache[cache_key] = (analyst, contrarian, consensus)
analyst, contrarian, consensus = st.session_state.ma_cache[cache_key]

st.markdown("### 🧠 Analyst Agent")
st.info(analyst)

st.markdown("### 🕵️ Contrarian Agent")
st.warning(contrarian)

st.markdown("### 🗳️ Consensus (Local Synthesis)")
st.success(consensus)

st.divider()
st.caption("Tip: Set env LLM_PROVIDER + LLM_MODEL_ID + API key. Example: GROQ + llama-3.1-8b-instruct, or OPENAI + gpt-4o-mini.")
