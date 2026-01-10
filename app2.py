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

# Brand 1 = The Fresh Market
BRAND_ID = "1"

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
        st.info("Please upload the raw data CSV.")
        st.stop()
    
    # Load data and strip any hidden spaces from column names
    raw_all = pd.read_csv(uploaded_file, low_memory=False)
    raw_all.columns = raw_all.columns.str.strip()

    # --- DYNAMIC COLUMN DISCOVERY ---
    # Look for RS3 (Awareness) for Brand 1
    # Could be RS3_1, RS3_1NET, RS3_brand1, etc.
    aware_col = next((c for c in raw_all.columns if c.startswith("RS3_") and (f"_{BRAND_ID}" in c or c.endswith(BRAND_ID))), None)
    
    # Look for RS8 (Salience) columns for Brand 1
    rs8_cols = [c for c in raw_all.columns if c.startswith("RS8_") and (f"_{BRAND_ID}" in c or c.endswith(BRAND_ID))]

    # --- DEBUG SIDEBAR ---
    with st.sidebar.expander("Column Debugger (Auto-Detected)"):
        st.write(f"Awareness Col: `{aware_col}`")
        st.write(f"Salience Cols Found: {len(rs8_cols)}")
        if st.checkbox("Show all column names?"):
            st.write(list(raw_all.columns))

    if not aware_col or not rs8_cols:
        st.error(f"Could not find required columns. Detected Awareness: {aware_col}, Salience: {len(rs8_cols)} cols.")
        st.stop()

    # --- FILTER & CALCULATE ---
    raw = raw_all[raw_all[MSA_FILTER_COL] == MSA_FILTER_VALUE].copy()
    w = raw[WEIGHT_COL].to_numpy()
    
    # Extract the CEP IDs from the RS8 column names
    # Expecting format RS8_CEPID_BRANDID (e.g., RS8_2_1)
    cep_data = []
    for col in rs8_cols:
        match = re.search(r"RS8_(\d+)_", col)
        if match:
            cep_id = int(match.group(1))
            
            # Math: Salience = (Sum of RS8=1 among RS3=1) / (Sum of Weights for RS3=1)
            aware_mask = (raw[aware_col] == 1).to_numpy()
            if aware_mask.sum() > 0:
                success = (raw[col] == 1).astype(int).to_numpy()
                salience = float(np.sum(success[aware_mask] * w[aware_mask]) / np.sum(w[aware_mask]))
                cep_data.append({"id": cep_id, "col": col, "baseline": salience})

    cep_df = pd.DataFrame(cep_data).sort_values("id")

    # --- SIMULATION INPUTS ---
    st.sidebar.header("Salience Uplift (Survey %)")
    uplifts = []
    for idx, row in cep_df.iterrows():
        name = CEP_NAME_MAP.get(row['id'], f"CEP {row['id']}")
        val = st.sidebar.slider(
            f"{name} (Baseline: {row['baseline']*100:.1f}%)",
            0, 25, 0, key=f"slider_{row['id']}"
        )
        uplifts.append(val / 100.0)

    # --- MONTE CARLO DEDUPLICATION ---
    # We build a matrix where 1 = household is 'reached' by that CEP
    n_rows = len(raw)
    X = np.zeros((n_rows, len(cep_df)), dtype=int)
    elig = np.zeros((n_rows, len(cep_df)), dtype=bool)
    
    for j, (idx, row) in enumerate(cep_df.iterrows()):
        X[:, j] = (raw[row['col']] == 1).astype(int).fillna(0).to_numpy()
        elig[:, j] = (raw[aware_col] == 1).to_numpy()

    # Probability Logic: Calculate unique reach
    # Prob(Reached) = 1 - Product(1 - Prob_i)
    # We apply the uplift proportionally to the 'unreached' eligible population
    s_current = cep_df['baseline'].values
    s_target = np.minimum(s_current + np.array(uplifts), 0.95)
    
    avg_aware = wmean((raw[aware_col] == 1).astype(int), w)
    
    # Deduplication math (Mental Penetration among the total market)
    def calc_penetration(salience_array):
        # The uniqueness factor represents the probability of a person having AT LEAST one association
        uniqueness = 1.0 - np.prod(1.0 - salience_array)
        return avg_aware * uniqueness

    current_mpen = calc_penetration(s_current)
    scenario_mpen = calc_penetration(s_target)

    # --- DISPLAY ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Mental Penetration", f"{current_mpen:,.1%}")
    c2.metric("Scenario Reach", f"{scenario_mpen:,.1%}", f"{(scenario_mpen - current_mpen):.1%}")
    
    gained_hh = (scenario_mpen - current_mpen) * HOUSEHOLD_BASE_TFM_STATES
    c3.metric("New Households Gained", f"{max(0, gained_hh):,.0f}")

    # Visualization
    chart_df = pd.DataFrame({
        "CEP": [CEP_NAME_MAP.get(r['id'], f"CEP {r['id']}") for _, r in cep_df.iterrows()],
        "Current": s_current * 100,
        "Scenario": s_target * 100
    }).sort_values("Current", ascending=False)
    
    st.subheader("CEP Salience Growth")
    st.bar_chart(chart_df.set_index("CEP"))

if __name__ == "__main__":
    main()
