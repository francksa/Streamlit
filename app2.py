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
AWARE_COL = "S2_1"  # Based on codebook: TFM Awareness

# ============================================================
# CEP NAME MAP
# ============================================================
CEP_NAME_MAP = {
    1: "Weekly grocery shopping",
    2: "Trying to save money",
    3: "Planning meals / supplies",
    4: "Errands + groceries",
    5: "Inspired to cook",
    6: "Healthy / organic foods",
    7: "Avoiding crowded stores",
    8: "Ready-to-eat meals",
    9: "Online / pickup shopping",
    10: "Exploring new flavors",
    11: "Better quality food",
    12: "Quick / convenient meal",
    13: "Special occasion",
    14: "Shopping for many",
    15: "Specialty / international",
    16: "Eco-friendly options",
    17: "Need help / ideas",
}

# ============================================================
# CORE MATH FUNCTIONS
# ============================================================

def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.sum(w))

def weighted_brand_salience(rs8_series: pd.Series, aware_series: pd.Series, w: np.ndarray) -> float:
    """
    ALIGNED TO DASHBOARD: Calculates salience among TOTAL BRAND AWARE respondents.
    1 = Selected TFM, 2 = Not selected, NaN = Not selected/asked.
    """
    aware_mask = (aware_series == 1).to_numpy()
    if aware_mask.sum() == 0:
        return 0.0
    
    # Success is strictly when TFM (brand 1) was selected
    success = (rs8_series == 1).astype(int).to_numpy()
    
    # Weighted calculation among the Aware base
    return float(np.sum(success[aware_mask] * w[aware_mask]) / np.sum(w[aware_mask]))

def simulate_unique_reach(X, elig, w, salience_current_w, uplifts_pts, n_sims, cap):
    """
    Monte Carlo Deduplication:
    Calculates how many UNIQUE households are reached as associations grow.
    """
    rng = np.random.default_rng(7)
    uplift = uplifts_pts / 100.0
    s_target = np.minimum(salience_current_w + uplift, cap)

    # Probability of 'flipping' a non-associator to an associator
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
        # Only flip if eligible, currently 0, and passes random check
        flips = elig & (X_sim == 0) & (U < flip_prob)
        X_sim[flips] = 1
        # Unique Reach: Does the respondent have AT LEAST ONE association?
        mpen_sim = (X_sim.max(axis=1) > 0).astype(int)
        reach_dist[t] = wmean(mpen_sim, w)

    return float(reach_dist.mean()), s_target

# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    st.title("TFM Mental Availability & TAM Simulator")

    # 1. Load Data
    uploaded_file = st.sidebar.file_uploader("Upload Raw Data CSV", type="csv")
    if uploaded_file is None:
        st.info("Please upload the raw data CSV to begin.")
        st.stop()
    
    raw_all = pd.read_csv(uploaded_file, low_memory=False)

    # 2. Filter for MSAs (Logic: xdemAud1 == 1)
    use_msa_only = st.sidebar.checkbox("Filter: TFM-present MSAs only", value=True)
    if use_msa_only:
        raw = raw_all[raw_all[MSA_FILTER_COL] == MSA_FILTER_VALUE].copy()
    else:
        raw = raw_all.copy()

    w = raw[WEIGHT_COL].to_numpy()
    
    # 3. Discover CEPs and Calculate Baselines
    cep_idx = [int(re.match(r"RS8_(\d+)_1NET", c).group(1)) 
               for c in raw.columns if re.match(r"RS8_(\d+)_1NET", c)]
    cep_idx.sort()

    # Fixed Salience: Base = Total Aware (S2_1)
    salience_w = np.array([
        weighted_brand_salience(raw[f"RS8_{cep}_1NET"], raw[AWARE_COL], w) 
        for cep in cep_idx
    ])

    # 4. Prepare Monte Carlo Inputs
    # X = matrix of current associations; elig = anyone who is AWARE (eligible to associate)
    n, k = len(raw), len(cep_idx)
    X = np.zeros((n, k), dtype=int)
    elig = np.zeros((n, k), dtype=bool)
    for j, cep in enumerate(cep_idx):
        X[:, j] = (raw[f"RS8_{cep}_1NET"] == 1).astype(int).to_numpy()
        elig[:, j] = (raw[AWARE_COL] == 1).to_numpy()

    # 5. Sidebar Sliders
    st.sidebar.header("Salience Uplift (Survey %)")
    uplifts = np.zeros(k)
    for j, cep in enumerate(cep_idx):
        label = CEP_NAME_MAP.get(cep, f"CEP {cep}")
        # This will now show the ~7.1% baseline from your dashboard
        uplifts[j] = st.sidebar.slider(
            f"{label} (Baseline: {salience_w[j]*100:.1f}%)",
            0, 20, 0
        )

    # 6. Run Simulation
    n_sims = st.sidebar.select_slider("Monte Carlo Runs", options=[200, 500, 1000], value=500)
    reach_scenario, target_salience = simulate_unique_reach(X, elig, w, salience_w, uplifts, n_sims, 0.9)

    # 7. Display Results
    # Current unique reach (un-simulated)
    current_reach = wmean((X.max(axis=1) > 0).astype(int), w)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Market Mental Penetration", f"{current_reach:,.1%}")
    col2.metric("Scenario Reach", f"{reach_scenario:,.1%}", f"{(reach_scenario - current_reach):.1%}")
    
    gained_hh = (reach_scenario - current_reach) * HOUSEHOLD_BASE_TFM_STATES
    col3.metric("New Households Gained", f"{max(0, gained_hh):,.0f}")

    # 8. Visualization
    st.subheader("CEP Salience Growth (Weighted Survey %)")
    chart_data = pd.DataFrame({
        "CEP": [CEP_NAME_MAP.get(c, f"CEP {c}") for c in cep_idx],
        "Current": salience_w * 100,
        "Scenario": target_salience * 100
    }).sort_values("Current", ascending=False)

    st.bar_chart(chart_data.set_index("CEP"))

if __name__ == "__main__":
    main()
