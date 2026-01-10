import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# 1. CONFIGURATION
# ============================================================
RAW_DATA_DEFAULT_NAMES = [
    "2510110_research_science_raw_data (1).csv",
    "2510110_research_science_raw_data.csv",
]

HOUSEHOLD_BASE_TFM_STATES = 70_132_819
MSA_FILTER_COL = "xdemAud1"
MSA_FILTER_VALUE = 1
WEIGHT_COL = "wts"
AWARE_COL = "RS3_1NET" 

DEFAULT_N_SIMS = 800
DEFAULT_CAP = 0.95
APP_DIR = Path(__file__).parent

CEP_NAME_MAP = {
    1: "Doing my regular weekly grocery shopping",
    2: "Trying to save money on groceries",
    3: "Planning meals and realizing I'm low on supplies",
    4: "Running other errands and picking up groceries too",
    5: "Feeling inspired to cook or try something new",
    6: "Looking for healthy or better-quality food options",
    7: "Avoiding crowded stores or long lines",
    8: "Buying ready-to-eat meals to save time",
    9: "Buying groceries online or for pickup",
    10: "Wanting to try new or seasonal products",
    11: "Feeling health-conscious and wanting to eat better",
    12: "Short on time but wanting convenient, good food",
    13: "Hosting guests or preparing for a special meal",
    14: "Shopping for a large family or group",
    15: "Looking for specialty or international foods",
    16: "Choosing environmentally friendly grocery options",
    17: "Needing help or ideas on what to buy",
}

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def wmean(x, wts):
    return float(np.sum(x * wts) / np.sum(wts))

def weighted_brand_salience(rs8_series, aware_series, wts):
    aware_mask = (aware_series == 1).to_numpy()
    if aware_mask.sum() == 0:
        return 0.0
    success = (rs8_series == 1).astype(int).to_numpy()
    return float(np.sum(success[aware_mask] * wts[aware_mask]) / np.sum(wts[aware_mask]))

def discover_valid_ceps(df):
    """Only returns CEP IDs that have both RS1 and RS8_brand1 columns."""
    valid_ids = []
    for i in range(1, 20): # Checking range 1-19
        rs1 = f"RS1_{i}NET"
        rs8 = f"RS8_{i}_1NET"
        if rs1 in df.columns and rs8 in df.columns:
            valid_ids.append(i)
    return valid_ids

def simulate_unique_reach(X, elig, wts, salience_current_w, uplifts_pts, n_sims, cap):
    rng = np.random.default_rng(7)
    uplift = uplifts_pts / 100.0
    s_target = np.minimum(salience_current_w + uplift, cap)
    
    denom = (1.0 - salience_current_w)
    need = (s_target - salience_current_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        flip_prob = np.where(denom > 1e-9, need / denom, 0.0)
    flip_prob = np.clip(flip_prob, 0.0, 1.0)
    
    n, k = X.shape
    reach_dist = np.zeros(n_sims, dtype=float)
    for t in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        flips = elig & (X_sim == 0) & (U < flip_prob)
        X_sim[flips] = 1
        mpen_sim = (X_sim.max(axis=1) > 0).astype(int)
        reach_dist[t] = wmean(mpen_sim, wts)
    return float(reach_dist.mean()), s_target

# ============================================================
# 3. MAIN APP
# ============================================================

def main():
    st.set_page_config(page_title="TFM Simulator", layout="wide")
    st.title("TFM Mental Availability & TAM Simulator")

    # Auto-load File
    raw_all = None
    for name in RAW_DATA_DEFAULT_NAMES:
        p = APP_DIR / name
        if p.exists():
            raw_all = pd.read_csv(p, low_memory=False)
            break

    if raw_all is None:
        st.error("Data file not found. Please ensure the CSV is in the app folder.")
        st.stop()

    # Apply MSA Filter
    raw = raw_all[raw_all[MSA_FILTER_COL] == MSA_FILTER_VALUE].copy()
    wts = raw[WEIGHT_COL].to_numpy()

    # Find valid CEPs (fixes the KeyError)
    cep_idx = discover_valid_ceps(raw)
    
    # Calculate Metrics (Awareness Base)
    salience_w = np.array([weighted_brand_salience(raw[f"RS8_{i}_1NET"], raw[AWARE_COL], wts) for i in cep_idx])
    prevalence_w = np.array([wmean((raw[f"RS1_{i}NET"] == 1).astype(int).to_numpy(), wts) for i in cep_idx])

    # Simulation Setup
    n, k = len(raw), len(cep_idx)
    X = np.zeros((n, k), dtype=int)
    elig = np.zeros((n, k), dtype=bool)
    aware_mask = (raw[AWARE_COL] == 1).to_numpy()
    for j, i in enumerate(cep_idx):
        X[:, j] = (raw[f"RS8_{i}_1NET"] == 1).astype(int).to_numpy()
        elig[:, j] = aware_mask

    # Sidebar
    st.sidebar.header("Salience Uplift (Aware Base %)")
    uplifts = np.zeros(k)
    order = np.argsort(-prevalence_w)
    for j in order:
        label = CEP_NAME_MAP.get(cep_idx[j], f"CEP {cep_idx[j]}")
        uplifts[j] = st.sidebar.slider(f"{label} (Baseline: {salience_w[j]*100:.1f}%)", 0, 50, 0)

    # Simulation
    reach_scenario, target_salience = simulate_unique_reach(X, elig, wts, salience_w, uplifts, DEFAULT_N_SIMS, DEFAULT_CAP)
    current_reach = wmean((X.max(axis=1) > 0).astype(int), wts)

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Mental Penetration", f"{current_reach:.1%}")
    c2.metric("Scenario Reach", f"{reach_scenario:.1%}", f"{(reach_scenario - current_reach):.1%}")
    c3.metric("New Households Gained", f"{(reach_scenario - current_reach) * HOUSEHOLD_BASE_TFM_STATES:,.0f}")

    # Results Table
    df = pd.DataFrame({
        "CEP": [CEP_NAME_MAP.get(i, f"CEP {i}") for i in cep_idx],
        "Category Prevalence %": prevalence_w * 100,
        "Current Brand Salience %": salience_w * 100,
        "Scenario Brand Salience %": target_salience * 100,
        "Uplift (pts)": uplifts,
    }).sort_values("Category Prevalence %", ascending=False)
    
    st.subheader("CEP Performance Metrics (MSA Respondents)")
    st.dataframe(df, use_container_width=True)

    # Bubble Chart
    st.subheader("Bubble Matrix: Growth Trail")
    base = alt.Chart(df).encode(x=alt.X("Category Prevalence %", title="Category Behavior Prevalence (%)"))
    trail = base.mark_rule(color="gray", strokeDash=[4,4], opacity=0.4).encode(
        y=alt.Y("Current Brand Salience %", title="TFM Salience among Aware (%)"),
        y2="Scenario Brand Salience %"
    )
    points = base.mark_circle(opacity=0.8, stroke="black", strokeWidth=1).encode(
        y="Current Brand Salience %",
        size=alt.Value(400),
        color=alt.value("#5b9244"),
        tooltip=["CEP", "Current Brand Salience %", "Scenario Brand Salience %"]
    )
    st.altair_chart((trail + points).properties(height=500), use_container_width=True)

if __name__ == "__main__":
    main()
