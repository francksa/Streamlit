import re
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# 1. DATA PATHS (Matches your uploaded files)
# ============================================================
RAW_DATA_FILE = "TFM_RAW_MSA.csv"
DASHBOARD_EXPORT = "Category Entry Points (CEPs).xlsx"

HOUSEHOLD_BASE_TFM_STATES = 70_132_819 
WEIGHT_COL = "wts"
AWARE_COL = "RS3_1NET" 

# ============================================================
# 2. DATA LOAD & ALIGNMENT
# ============================================================

def load_and_align_data():
    # Load Dashboard Truths
    # Note: Header is usually further down in these exports, we search for the labels
    dash_df = pd.read_csv(DASHBOARD_EXPORT, encoding='utf-8-sig')
    
    # Load Raw Respondent Data
    raw_df = pd.read_csv(RAW_DATA_FILE, encoding='utf-8-sig', low_memory=False)
    raw_df[WEIGHT_COL] = pd.to_numeric(raw_df[WEIGHT_COL], errors='coerce').fillna(0)
    
    # Extract Baselines from Dashboard File
    # We look for rows that match our CEP labels
    baselines = []
    
    # Mapping table for the simulator
    mapping = {
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

    for cid, label in mapping.items():
        # Find the value in the dashboard export
        # We look for the row where the first column matches the label
        row = dash_df[dash_df.iloc[:, 0].str.contains(label, na=False, case=False)]
        if not row.empty:
            val = float(row.iloc[0, 1]) # The % value
            # We also need the Category Prevalence (usually found in a different section of that file)
            # For this simulator, we will derive prevalence from the raw data to ensure TAM consistency
            cat_prev = (raw_df[f"RS1_{cid}NET"] == 1).mean() 
            
            baselines.append({
                "id": cid,
                "label": label,
                "salience": val,
                "prevalence": cat_prev
            })
            
    return pd.DataFrame(baselines), raw_df

# ============================================================
# 3. SIMULATION ENGINE
# ============================================================

def run_simulation(raw_df, cep_df, uplifts, n_sims=800):
    rng = np.random.default_rng(7)
    wts = raw_df[WEIGHT_COL].to_numpy()
    
    # Build current association matrix
    n, k = len(raw_df), len(cep_df)
    X = np.zeros((n, k), dtype=int)
    elig = (raw_df[AWARE_COL] == 1).to_numpy()
    
    current_saliences = cep_df['salience'].values
    target_saliences = np.minimum(current_saliences + (uplifts / 100.0), 0.95)
    
    for j, cid in enumerate(cep_df['id']):
        X[:, j] = (raw_df[f"RS8_{cid}_1NET"] == 1).astype(int).to_numpy()

    # Probability to 'flip' a household to TFM
    denom = (1.0 - current_saliences)
    need = (target_saliences - current_saliences)
    flip_probs = np.where(denom > 0, need / denom, 0.0)

    sim_mpens = []
    for _ in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        # Flip Aware respondents who don't already associate TFM with that CEP
        flips = (X_sim == 0) & (elig[:, None]) & (U < flip_probs)
        X_sim[flips] = 1
        # Reach = respondent has at least one 1
        sim_mpens.append(np.sum((X_sim.max(axis=1) > 0) * wts) / np.sum(wts))
        
    return float(np.mean(sim_mpens)), target_saliences

# ============================================================
# 4. STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="TFM Final Simulator")
    st.title("TFM Mental Availability & TAM Simulator")
    st.info("Aligned to Dashboard Export: Healthy Food @ ~34%, Saving Money @ ~7.5%")

    try:
        cep_df, raw_df = load_and_align_data()
    except Exception as e:
        st.error(f"Error aligning data: {e}")
        st.stop()

    # Sidebar
    st.sidebar.header("Salience Uplift (pts)")
    uplifts = []
    # Sort by prevalence to show most important CEPs first
    for idx, row in cep_df.sort_values("prevalence", ascending=False).iterrows():
        val = st.sidebar.slider(f"{row['label']} (Base: {row['salience']:.1%})", 0, 25, 0, key=f"s_{row['id']}")
        uplifts.append((row['id'], val))
    
    # Run Simulation
    u_array = np.array([u[1] for u in sorted(uplifts, key=lambda x: x[0])])
    scenario_mpen, target_saliences = run_simulation(raw_df, cep_df, u_array)
    current_mpen = np.sum(((raw_df.filter(regex='RS8_.*_1NET') == 1).max(axis=1)) * raw_df[WEIGHT_COL]) / raw_df[WEIGHT_COL].sum()

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Mental Penetration", f"{current_mpen:.1%}")
    c2.metric("Scenario Reach", f"{scenario_mpen:.1%}", f"{(scenario_reach - current_reach):+.1%}" if 'scenario_reach' in locals() else None)
    c3.metric("New Households Gained", f"{(scenario_mpen - current_mpen) * HOUSEHOLD_BASE_TFM_STATES:,.0f}")

    # Results Table
    results_df = cep_df.copy()
    results_df['Scenario'] = target_saliences
    st.subheader("CEP Performance Detail")
    st.dataframe(results_df[['label', 'salience', 'Scenario']].style.format(formatter="{:.1%}"), use_container_width=True)

if __name__ == "__main__":
    main()
