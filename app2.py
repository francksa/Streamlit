import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# CONFIG & CONSTANTS
# ============================================================
HOUSEHOLD_BASE_TFM_STATES = 70_132_819
MSA_FILTER_COL = "xdemAud1"
MSA_FILTER_VALUE = 1
WEIGHT_COL = "wts"

# THE KEY FIX: RS3_1 is the column for TFM Awareness
AWARE_COL = "RS3_1" 

CEP_NAME_MAP = {
    1: "Weekly grocery shopping",
    2: "Trying to save money",
    3: "Planning meals / supplies",
    4: "Errands + groceries",
    5: "Inspired to cook",
    6: "Exploring new flavors",
    11: "Healthy / organic foods",
    12: "Quick / convenient meal",
    13: "Special occasion",
    15: "Specialty / international",
}

# ============================================================
# CORE MATH FUNCTIONS
# ============================================================

def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))

def weighted_brand_salience(rs8_series, aware_series, w):
    """
    Calculates salience among those who said 'Yes' to RS3_1 (Awareness).
    """
    # Base: People who are aware (RS3_1 == 1)
    aware_mask = (aware_series == 1).to_numpy()
    
    if aware_mask.sum() == 0:
        return 0.0
    
    # Success: People who associate TFM with this CEP (RS8_i_1 == 1)
    # We use .fillna(0) because if they weren't asked, they don't associate.
    success = (rs8_series == 1).astype(int).to_numpy()
    
    return float(np.sum(success[aware_mask] * w[aware_mask]) / np.sum(w[aware_mask]))

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.set_page_config(layout="wide", page_title="TFM Simulator")
    st.title("TFM Mental Availability & TAM Simulator")

    uploaded_file = st.sidebar.file_uploader("Upload Raw Data CSV", type="csv")
    if not uploaded_file:
        st.info("Please upload the raw data CSV.")
        st.stop()
    
    raw_all = pd.read_csv(uploaded_file, low_memory=False)

    # Validate AWARE_COL
    if AWARE_COL not in raw_all.columns:
        st.error(f"Critical Column Missing: {AWARE_COL} (TFM Awareness).")
        st.stop()

    # Apply MSA Filter
    raw = raw_all[raw_all[MSA_FILTER_COL] == MSA_FILTER_VALUE].copy()
    w = raw[WEIGHT_COL].to_numpy()
    
    # Discover CEPs based on RS8 columns
    # Your columns might look like RS8_1_1, RS8_2_1, etc.
    rs8_pattern = r"RS8_(\d+)_1" 
    rs8_cols = [c for c in raw.columns if re.match(rs8_pattern, c)]
    cep_idx = sorted([int(re.search(rs8_pattern, c).group(1)) for c in rs8_cols])

    # Calculate Baselines (Salience among the Aware)
    salience_w = np.array([
        weighted_brand_salience(raw[f"RS8_{cep}_1"], raw[AWARE_COL], w) 
        for cep in cep_idx
    ])

    # Simulation Setup
    n_rows = len(raw)
    X = np.zeros((n_rows, len(cep_idx)), dtype=int)
    elig = np.zeros((n_rows, len(cep_idx)), dtype=bool)
    
    for j, cep in enumerate(cep_idx):
        X[:, j] = (raw[f"RS8_{cep}_1"] == 1).astype(int).fillna(0).to_numpy()
        elig[:, j] = (raw[AWARE_COL] == 1).to_numpy()

    # Sidebar
    st.sidebar.header("Salience Uplift (Survey %)")
    uplifts = np.zeros(len(cep_idx))
    for j, cep in enumerate(cep_idx):
        name = CEP_NAME_MAP.get(cep, f"CEP {cep}")
        uplifts[j] = st.sidebar.slider(
            f"{name} (Baseline: {salience_w[j]*100:.1f}%)",
            0, 25, 0
        )

    # Run Monte Carlo
    # (Using simplified reach logic for performance)
    s_target = np.minimum(salience_w + (uplifts / 100.0), 0.95)
    
    # Calculate unique reach: 1 - Prob(Not knowing any CEP)
    # This is the 'Theoretical' deduplication based on probability
    current_uniqueness_factor = 1.0 - np.prod(1.0 - salience_w)
    scenario_uniqueness_factor = 1.0 - np.prod(1.0 - s_target)
    
    # Market Penetration = Awareness * Uniqueness Factor
    avg_awareness = wmean((raw[AWARE_COL] == 1).astype(int), w)
    current_reach = avg_awareness * current_uniqueness_factor
    scenario_reach = avg_awareness * scenario_uniqueness_factor
    
    # Display Results
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Mental Penetration", f"{current_reach:,.1%}")
    c2.metric("Scenario Reach", f"{scenario_reach:,.1%}", f"{(scenario_reach - current_reach):.1%}")
    
    gained_hh = (scenario_reach - current_reach) * HOUSEHOLD_BASE_TFM_STATES
    c3.metric("New Households Gained", f"{max(0, gained_hh):,.0f}")

    # Chart
    chart_df = pd.DataFrame({
        "CEP": [CEP_NAME_MAP.get(c, f"CEP {c}") for c in cep_idx],
        "Current": salience_w * 100,
        "Scenario": s_target * 100
    }).sort_values("Current", ascending=False)
    
    st.subheader("CEP Salience: Current vs Scenario")
    st.bar_chart(chart_df.set_index("CEP"))

if __name__ == "__main__":
    main()
