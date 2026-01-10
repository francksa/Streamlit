import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# CONFIG
# ============================================================
HOUSEHOLD_BASE_TFM_STATES = 70_132_819
MSA_FILTER_COL = "xdemAud1"
MSA_FILTER_VALUE = 1
WEIGHT_COL = "wts"
BRAND_ID = "1" # TFM

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

def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.set_page_config(layout="wide", page_title="TFM Simulator")
    st.title("TFM Mental Availability & TAM Simulator")

    uploaded_file = st.sidebar.file_uploader("Upload Raw Data CSV", type="csv")
    if not uploaded_file:
        st.info("Please upload the raw data CSV to begin.")
        st.stop()
    
    raw_all = pd.read_csv(uploaded_file, low_memory=False)
    raw_all.columns = raw_all.columns.str.strip()

    # --- DYNAMIC COLUMN DISCOVERY ---
    # Find Awareness (RS3) for TFM
    aware_col = next((c for c in raw_all.columns if c.startswith("RS3_") and (f"_{BRAND_ID}" in c or c.endswith(BRAND_ID))), None)
    
    # Find Salience (RS8) columns for TFM
    rs8_cols = [c for c in raw_all.columns if c.startswith("RS8_") and (f"_{BRAND_ID}" in c or c.endswith(BRAND_ID))]

    if not aware_col or not rs8_cols:
        st.error(f"Missing Columns. Found Aware: {aware_col}, Salience Columns: {len(rs8_cols)}")
        st.stop()

    # --- FILTER & CALCULATE ---
    raw = raw_all[raw_all[MSA_FILTER_COL] == MSA_FILTER_VALUE].copy()
    w = raw[WEIGHT_COL].to_numpy()
    
    # DEDUPLICATION LOGIC: Ensure only one column per CEP ID
    cep_dict = {}
    for col in rs8_cols:
        match = re.search(r"RS8_(\d+)_", col)
        if match:
            cep_id = int(match.group(1))
            # If we have multiple columns for one CEP, prefer the one with 'NET' in the name
            if cep_id not in cep_dict or "NET" in col:
                cep_dict[cep_id] = col

    # Build the final list of CEPs for the simulation
    cep_data = []
    aware_mask = (raw[aware_col] == 1).to_numpy()
    
    for cep_id, col in cep_dict.items():
        if aware_mask.sum() > 0:
            success = (raw[col] == 1).astype(int).to_numpy()
            salience = float(np.sum(success[aware_mask] * w[aware_mask]) / np.sum(w[aware_mask]))
            cep_data.append({"id": cep_id, "col": col, "baseline": salience})

    cep_df = pd.DataFrame(cep_data).sort_values("id")

    # --- SIDEBAR SLIDERS ---
    st.sidebar.header("Salience Uplift (Survey %)")
    uplift_values = []
    for idx, row in cep_df.iterrows():
        name = CEP_NAME_MAP.get(row['id'], f"CEP {row['id']}")
        # Unique key using both ID and name to avoid DuplicateElementKey error
        u_val = st.sidebar.slider(
            f"{name} (Baseline: {row['baseline']*100:.1f}%)",
            min_value=0, max_value=25, value=0,
            key=f"slider_cep_{row['id']}_{idx}" 
        )
        uplift_values.append(u_val / 100.0)

    # --- SIMULATION LOGIC ---
    s_current = cep_df['baseline'].values
    s_target = np.minimum(s_current + np.array(uplift_values), 0.95)
    
    # Average Awareness in the MSA footprint
    avg_aware = wmean((raw[aware_col] == 1).astype(int), w)
    
    # Deduplication Formula: 1 - Product(1 - Prob_i)
    # This accounts for the 'Same Household' overlap
    def get_market_reach(salience_probs):
        uniqueness_factor = 1.0 - np.prod(1.0 - salience_probs)
        return avg_aware * uniqueness_factor

    current_reach = get_market_reach(s_current)
    scenario_reach = get_market_reach(s_target)

    # --- RESULTS DASHBOARD ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Mental Penetration", f"{current_reach:,.1%}")
    c2.metric("Scenario Reach", f"{scenario_reach:,.1%}", f"{(scenario_reach - current_reach):,.1%}")
    
    gained_hh = (scenario_reach - current_reach) * HOUSEHOLD_BASE_TFM_STATES
    c3.metric("New Households Gained", f"{max(0, gained_hh):,.0f}")

    # --- CHART ---
    chart_df = pd.DataFrame({
        "CEP": [CEP_NAME_MAP.get(r['id'], f"CEP {r['id']}") for _, r in cep_df.iterrows()],
        "Current": s_current * 100,
        "Scenario": s_target * 100
    }).sort_values("Current", ascending=False)
    
    st.subheader("Mental Availability Growth by Category Entry Point")
    st.bar_chart(chart_df.set_index("CEP"))

if __name__ == "__main__":
    main()
