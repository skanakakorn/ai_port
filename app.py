
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Portfolio Agent", page_icon="🤖")

st.title("🤖 AI Portfolio Agent — Demo")
st.caption("Game-changing: Let your personal AI construct portfolios by theme and risk profile.")

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

# --- Portfolio constructors ---
def cap_weights(w, cap):
    w = np.array(w, dtype=float)
    if w.sum() == 0:
        return w
    w = w / w.sum()
    # simple capping then renormalize loop
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
    # pick N lowest vol
    pick = df.nsmallest(n, "volatility").copy()
    # risk parity-ish: weight ~ 1/vol
    inv = 1.0 / np.maximum(pick["volatility"].values, 1e-6)
    w = inv / inv.sum()
    w = cap_weights(w, max_weight)
    pick["weight"] = w
    pick["strategy"] = "A) Defensive Low-Vol Infra"
    return pick

def portfolio_balanced(df, n):
    # pick by expected_return / volatility (Sharpe proxy)
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
    # top N by momentum_6m, conviction capped
    pick = df.nlargest(n, "momentum_6m").copy()
    # rescale momentum to positive weights
    m = pick["momentum_6m"].values - pick["momentum_6m"].min() + 1e-6
    w = m / m.sum()
    w = cap_weights(w, max_weight)
    pick["weight"] = w
    pick["strategy"] = "C) High-Conviction Momentum"
    return pick

# Build three contrasting portfolios
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
    vol = (df["volatility"].values * w).sum()  # heuristic stand-in for demo
    return er, vol

st.subheader("Portfolios")
for label, pf in result.groupby("strategy"):
    st.markdown(f"**{label}**")
    st.dataframe(
        pf[["ticker","name","theme","expected_return","volatility","momentum_6m","weight"]]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True),
        use_container_width=True, hide_index=True
    )
    exp_r, vol = summarize_portfolio(pf)
    st.write(f"Expected Return (heuristic): **{exp_r:.2%}**,  Volatility (heuristic): **{vol:.2%}**")

    # Single chart per portfolio (matplotlib, no specific colors)
    fig, ax = plt.subplots()
    ax.bar(pf["ticker"], pf["weight"])
    ax.set_title(f"Weights — {label}")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Weight")
    st.pyplot(fig)

# Export
st.subheader("Export")
def to_csv_bytes(df):
    out = io.StringIO()
    df.to_csv(out, index=False)
    return out.getvalue().encode()

st.download_button("Download All Portfolios (CSV)", data=to_csv_bytes(result), file_name="ai_portfolios.csv", mime="text/csv")

# Narrative (no external LLM calls; template-based)
st.subheader("Narrative (for stage script)")
def make_narrative(df):
    lines = []
    for label, pf in df.groupby("strategy"):
        er, vol = summarize_portfolio(pf)
        top3 = ", ".join(pf.sort_values("weight", ascending=False).head(3)["ticker"].tolist())
        lines.append(f"{label}: Targets ~{er:.1%} ER with ~{vol:.1%} demo-vol; top names: {top3}.")
    return "\n".join(lines)

st.text_area("Auto summary", value=make_narrative(result), height=120)
