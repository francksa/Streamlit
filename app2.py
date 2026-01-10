import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# ============================================================
# CONFIG & CONSTANTS
# ============================================================
HOUSEHOLD_BASE_TFM_STATES = 70_132_819
MSA_FILTER_COL = "xdemAud1"
MSA_FILTER_VALUE = 1
WEIGHT_COL = "wts"

# Common names for the Awareness column in TFM datasets
AWARE_CANDIDATES = ["S2_1", "S2_1NET", "S2_Brand1", "Awareness_1"]

# ============================================================
# CEP NAME MAP (Updated for your Dashboard)
# ============================================================
CEP_NAME_MAP = {
    1: "Weekly grocery shopping",
    2: "Trying to save money",
    3: "Planning meals / supplies",
    4: "Errands + groceries",
    5: "Inspired to cook",
    6: "Exploring new flavors",
    11: "Healthy / organic foods",
    12: "Quick / convenient meal",
    # Add others as needed to match your screenshot
}

# ============================================================
# CORE MATH FUNCTIONS
# ============================================================

def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))

def weighted_brand_salience(rs8_series, aware_series, w):
    """
    ALIGNED TO DASHBOARD: Base = Total Brand Aware respondents in MSA.
    """
    # 1 = Aware, everything else is not
    aware_mask = (aware_series == 1).to_numpy()
    if aware_mask.sum() == 0:
        return 0.0
    
    # 1 = Selected TFM for CEP
    success = (rs8_series == 1).astype(int).to_numpy()
    return float(np.sum(success[aware_mask] * w[aware_mask]) / np.sum(w[aware_mask]))

def simulate_unique_reach(X, elig, w, salience_current_w, uplifts_pts, n_sims, cap):
    rng = np.random.default_rng(7)
    uplift = uplifts_pts / 100.0
    s_target = np.minimum(salience_current_w + uplift, cap)

    denom = (1.0 - salience_current_w)
    need = (s_target - salience_current_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        flip_prob = np.where(denom > 1e-9, need / denom, 0.0)
    flip_prob = np.clip(flip_prob, 0.0, 1.0)

    n, k = X.shape
    reach_dist = np.zeros(n_sims)

    for t in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        flips = elig & (X_sim == 0) & (U < flip_prob)
        X_sim[flips] = 1
        mpen_sim = (X_sim.max(axis=1) > 0).astype(int)
        reach_dist[t] = wmean(mpen_sim, w)

    return float(reach_dist.mean()), s_target

# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    st.set_page_config(layout="wide")
    st.title("TFM Mental Availability & TAM Simulator")

    uploaded_file = st.sidebar.file_uploader("Upload Raw Data CSV", type="csv")
    if uploaded_file is None:
        st.info("Please upload the raw data CSV to begin.")
        st.stop()
    
    raw_all = pd.read_csv(uploaded_file, low_memory=False)

    # --- Awareness Column Discovery ---
    aware_col = next((c for c in AWARE_CANDIDATES if c in raw_all.columns), None)
    if not aware_col:
        st.error("Could not find Awareness column (S2_1). Please select it below:")
        aware_col = st.selectbox("Select Awareness Column", options=raw_all.columns)

    # --- MSA Filter ---
    raw = raw_all[raw_all[MSA_FILTER_COL] == MSA_FILTER_VALUE].copy()
    w = raw[WEIGHT_COL].to_numpy()
    
    # --- CEP Calculation ---
    # Find all RS8 columns for Brand 1 (TFM)
    rs8_cols = [c for c in raw.columns if re.match(r"RS8_(\d+)_1NET", c)]
    cep_idx = sorted([int(re.search(r"RS8_(\d+)_1NET", c).group(1)) for c in rs8_cols])

    # Calculate Weighted Salience among AWARE base
    salience_w = np.array([
        weighted_brand_salience(raw[f"RS8_{cep}_1NET"], raw[aware_col], w) 
        for cep in cep_idx
    ])

    # --- Simulation Setup ---
    X = np.zeros((len(raw), len(cep_idx)), dtype=int)
    elig = np.zeros((len(raw), len(cep_idx)), dtype=bool)
    for j, cep in enumerate(cep_idx):
        X[:, j] = (raw[f"RS8_{cep}_1NET"] == 1).astype(int).to_numpy()
        elig[:, j] = (raw[aware_col] == 1).to_numpy()

    # --- Sidebar ---
    st.sidebar.header("Salience Uplift (Survey %)")
    uplifts = np.zeros(len(cep_idx))
    for j, cep in enumerate(cep_idx):
        name = CEP_NAME_MAP.get(cep, f"CEP {cep}")
        # Logic check: 'Trying to save money' should now show ~7.1% as baseline
        uplifts[j] = st.sidebar.slider(
            f"{name} (Baseline: {salience_w[j]*100:.1f}%)",
            0, 20, 0
        )

    # --- Run & Display ---
    n_sims = 500
    reach_scenario, target_salience = simulate_unique_reach(X, elig, w, salience_w, uplifts, n_sims, 0.9)
    current_reach = wmean((X.max(axis=1) > 0).astype(int), w)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Mental Penetration", f"{current_reach:,.1%}")
    c2.metric("Scenario Reach", f"{reach_scenario:,.1%}", f"{(reach_scenario - current_reach):.1%}")
    
    gained_hh = (reach_scenario - current_reach) * HOUSEHOLD_BASE_TFM_STATES
    c3.metric("New Households Gained", f"{max(0, gained_hh):,.0f}")

    # --- Chart ---
    chart_df = pd.DataFrame({
        "CEP": [CEP_NAME_MAP.get(c, f"CEP {c}") for c in cep_idx],
        "Current %": salience_w * 100,
        "Scenario %": target_salience * 100
    }).sort_values("Current %", ascending=True)

    # Using horizontal bars to match the dashboard style
    base = alt.Chart(chart_df).encode(y=alt.Y("CEP", sort='-x'))
    bars = base.mark_bar(color="#5b9244").encode(x=alt.X("Current %", title="Salience %"))
    st.altair_chart(bars, use_container_width=True)

if __name__ == "__main__":
    main()
