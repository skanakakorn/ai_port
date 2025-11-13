
import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------- LLM Provider Adaptor (choose one in env) ----------
# Supported providers: OPENAI, TOGETHER, GROQ, (or dummy)
# Set:
#   LLM_PROVIDER = "OPENAI" | "TOGETHER" | "GROQ" | "DUMMY"
#   LLM_MODEL_ID = (e.g., "gpt-4o-mini" or "llama-3.1-8b-instruct" etc.)
#   API keys:
#     OPENAI_API_KEY / TOGETHER_API_KEY / GROQ_API_KEY

def call_llm_cheap(prompt, system=None, temperature=0.2, max_tokens=400):
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
                model=model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM error OPENAI: {e}]"

    elif provider == "TOGETHER":
        try:
            import requests
            url = "https://api.together.xyz/v1/chat/completions"
            headers = {"Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
                       "Content-Type": "application/json"}
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            payload = {
                "model": model,
                "messages": msgs,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
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
            headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                       "Content-Type": "application/json"}
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            payload = {
                "model": model,
                "messages": msgs,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            try:
                r.raise_for_status()
            except requests.HTTPError as http_err:
                # Include response text for clarity (e.g., model_not_found)
                detail = r.text.strip() if hasattr(r, "text") else ""
                return f"[LLM error GROQ: {r.status_code} {http_err} {detail}]"
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[LLM error GROQ: {e}]"

    else:
        # Safe fallback for dev/demo without keys
        return ("[DUMMY LLM]\n"
                "This is a placeholder response. Set LLM_PROVIDER and API key "
                "to get real answers.")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Portfolio Agent + Cheap LLM Explainer", page_icon="🤖")

st.title("🤖 AI Portfolio Agent — with Low-cost LLM Stock Explainer")
st.caption("Construct portfolios and chat with a budget LLM to understand the picks.")

st.sidebar.header("1) Load Universe Data")
uploaded = st.sidebar.file_uploader("Upload CSV (ticker,name,theme,mcap_bucket,expected_return,volatility,momentum_6m)", type=["csv"])

@st.cache_data
def load_default():
    return pd.read_csv("ai_universe_sample.csv")

if uploaded is not None:
    universe = pd.read_csv(uploaded)
else:
    st.sidebar.info("Using bundled sample: ai_universe_sample.csv")
    universe = load_default()

# Basic sanity
needed_cols = {"ticker","name","theme","mcap_bucket","expected_return","volatility","momentum_6m"}
if not needed_cols.issubset(set(universe.columns)):
    st.error(f"CSV must include columns: {sorted(list(needed_cols))}")
    st.stop()

st.sidebar.header("2) Strategy Controls")
n_names = st.sidebar.slider("Stocks per portfolio", min_value=6, max_value=20, value=10, step=1)
max_weight = st.sidebar.slider("Max weight per name", min_value=0.05, max_value=0.3, value=0.15, step=0.01)
risk_floor = st.sidebar.slider("Volatility floor for selection (optional)", min_value=0.0, max_value=0.6, value=0.0, step=0.01)

# Optional theme filters
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

# --- Portfolio builders ---
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

def portfolio_low_vol(df, n):
    pick = df.nsmallest(n, "volatility").copy()
    inv = 1.0 / np.maximum(pick["volatility"].values, 1e-6)
    w = inv / inv.sum()
    w = cap_weights(w, max_weight)
    pick["weight"] = w
    pick["strategy"] = "A) Defensive Low-Vol Infra"
    return pick

def portfolio_balanced(df, n):
    df = df.copy()
    df["score"] = df["expected_return"] / np.maximum(df["volatility"], 1e-6)
    pick = df.nlargest(n, "score").copy()
    w = pick["score"].values
    w = w / w.sum()
    w = cap_weights(w, max_weight)
    pick["weight"] = w
    pick["strategy"] = "B) Balanced Sharpe Mix"
    return pick.drop(columns=["score"])

def portfolio_momentum(df, n):
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

pA = portfolio_low_vol(base, n_names)
pB = portfolio_balanced(base, n_names)
pC = portfolio_momentum(base, n_names)
result = pd.concat([pA, pB, pC], axis=0, ignore_index=True)

def summarize_portfolio(df):
    w = df["weight"].values
    er = (df["expected_return"].values * w).sum()
    vol = (df["volatility"].values * w).sum()  # heuristic
    return er, vol

st.subheader("Portfolios")
for label, pf in result.groupby("strategy"):
    st.markdown(f"**{label}**")
    st.dataframe(
        pf[["ticker","name","theme","expected_return","volatility","momentum_6m","weight"]]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True) if hasattr(pd.DataFrame, 'reset_index') else pf,
        use_container_width=True, hide_index=True
    )
    exp_r, vol = summarize_portfolio(pf)
    st.write(f"Expected Return (heuristic): **{exp_r:.2%}**,  Volatility (heuristic): **{vol:.2%}**")

    fig, ax = plt.subplots()
    ax.bar(pf["ticker"], pf["weight"])
    ax.set_title(f"Weights — {label}")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Weight")
    st.pyplot(fig)

# ---------- Export ----------
def to_csv_bytes(df):
    out = io.StringIO()
    df.to_csv(out, index=False)
    return out.getvalue().encode()

st.download_button("Download All Portfolios (CSV)", data=to_csv_bytes(result), file_name="ai_portfolios.csv", mime="text/csv")

# ---------- Narrative ----------
st.subheader("Narrative (for stage script)")
def make_narrative(df):
    lines = []
    for label, pf in df.groupby("strategy"):
        er, vol = summarize_portfolio(pf)
        top3 = ", ".join(pf.sort_values("weight", ascending=False).head(3)["ticker"].tolist())
        lines.append(f"{label}: Targets ~{er:.1%} ER with ~{vol:.1%} demo-vol; top names: {top3}.")
    return "\n".join(lines)
st.text_area("Auto summary", value=make_narrative(result), height=120)

# ---------- Cheap LLM Stock Explainer ----------
st.subheader("💬 Stock Explainer (Low-cost LLM)")
st.caption("Select a picked stock and ask questions. Uses a budget LLM provider you set via environment variables.")

# Build a quick lookup for picked tickers only
picked = result[["ticker","name","theme","mcap_bucket","expected_return","volatility","momentum_6m","strategy","weight"]].copy()
picked_tickers = picked["ticker"].unique().tolist()
if len(picked_tickers) == 0:
    st.info("No picks available to chat about yet.")
else:
    st.write("Choose a ticker from the constructed portfolios.")
    chosen = st.selectbox("Ticker", picked_tickers)
    row = picked[picked["ticker"] == chosen].iloc[0].to_dict()

    # Show context card
    st.markdown(f"**{row['ticker']} — {row['name']}**")
    st.markdown(f"- Theme: {row['theme']}  \n- Market-cap: {row['mcap_bucket']}  \n- In strategy: {row['strategy']}  \n- Weight: {row['weight']:.1%}")
    st.markdown(f"- Expected return: {row['expected_return']:.1%}  \n- Volatility: {row['volatility']:.1%}  \n- 6M momentum: {row['momentum_6m']:.1%}")

    # Simple chat state per ticker
    if "chat" not in st.session_state:
        st.session_state.chat = {}
    if chosen not in st.session_state.chat:
        st.session_state.chat[chosen] = []

    def render_chat():
        for role, content in st.session_state.chat[chosen]:
            if role == "user":
                st.chat_message("user").markdown(content)
            else:
                st.chat_message("assistant").markdown(content)

    render_chat()
    user_msg = st.chat_input("Ask about this stock (e.g., 'Why is it in the low-vol portfolio?', 'Key risks?')")

    if user_msg:
        st.session_state.chat[chosen].append(("user", user_msg))

        # Minimal, cheap prompt with tight context to keep tokens low
        system_prompt = (
            "You are a concise equity explainer for a portfolio agent. "
            "Use ONLY the provided facts; do not fabricate data, prices, or news. "
            "If the user asks for anything beyond provided facts, answer with general principles and clearly say 'no real-time data here'."
        )
        compact_context = (
            f"Ticker: {row['ticker']}\n"
            f"Name: {row['name']}\n"
            f"Theme: {row['theme']}\n"
            f"MarketCapBucket: {row['mcap_bucket']}\n"
            f"Strategy: {row['strategy']}\n"
            f"Weight: {row['weight']:.3f}\n"
            f"ExpectedReturn: {row['expected_return']:.3f}\n"
            f"Volatility: {row['volatility']:.3f}\n"
            f"Momentum6M: {row['momentum_6m']:.3f}\n"
        )
        prompt = (
            "Context:\n" + compact_context +
            "\nTask: Answer the user's question briefly (<= 160 words), "
            "linking the stock's role to the selected strategy and highlighting 1–2 risks/trade-offs. "
            "Avoid real-time claims.\n\n"
            f"User question: {user_msg}"
        )

        reply = call_llm_cheap(prompt, system=system_prompt, temperature=0.2, max_tokens=300)
        st.session_state.chat[chosen].append(("assistant", reply))
        render_chat()

st.divider()
# st.caption("Tip: Set env vars LLM_PROVIDER, LLM_MODEL_ID, and the corresponding API key for real answers. Example: OPENAI + gpt-4o-mini, or GROQ + llama-3.1-8b-instruct.")
