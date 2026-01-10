import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# 1. CONFIGURATION
# ============================================================
RAW_DATA_FILE = "2510110_research_science_raw_data (1).csv"
HOUSEHOLD_BASE_TFM_STATES = 70_132_819 # Total TAM
MSA_FILTER_COL = "xdemAud1"
MSA_FILTER_VALUE = 1
WEIGHT_COL = "wts"
AWARE_COL = "RS3_1NET" # TFM Awareness

CEP_NAME_MAP = {
    1: "Weekly grocery shopping",
    2: "Trying to save money on groceries",
    3: "Planning meals / low on supplies",
    4: "Running other errands",
    5: "Inspired to cook / try something new",
    6: "Healthy / better-quality options",
    7: "Avoiding crowded stores",
    8: "Ready-to-eat meals",
    9: "Online / pickup shopping",
    10: "New or seasonal products",
    11: "Health-conscious / eat better",
    12: "Convenient, good food",
    13: "Special meal / hosting",
    14: "Large family / group",
    15: "Specialty / international foods",
    16: "Eco-friendly options",
    17: "Help or ideas",
}

# ============================================================
# 2. CALCULATION ENGINE
# ============================================================

def get_msa_baselines(df):
    """Calculates salience % ONLY for MSA respondents who are Aware of TFM."""
    # Filter for MSA and Weights
    msa_df = df[df[MSA_FILTER_COL] == MSA_FILTER_VALUE].copy()
    w = msa_df[WEIGHT_COL].to_numpy()
    
    # Calculate Awareness in MSA
    aware_mask = (msa_df[AWARE_COL] == 1).to_numpy()
    
    cep_results = []
    for i in range(1, 18):
        rs1_col = f"RS1_{i}NET"
        rs8_col = f"RS8_{i}_1NET"
        
        if rs1_col in msa_df.columns and rs8_col in msa_df.columns:
            # Category Prevalence (Total Market in MSA)
            cat_prev = np.sum((msa_df[rs1_col] == 1) * w) / np.sum(w)
            
            # Brand Salience (Among Aware in MSA)
            brand_sal = 0.0
            if aware_mask.sum() > 0:
                brand_sal = np.sum((msa_df[rs8_col] == 1) * w[aware_mask]) / np.sum(w[aware_mask])
            
            cep_results.append({
                "id": i,
                "label": CEP_NAME_MAP.get(i, f"CEP {i}"),
                "prevalence": cat_prev,
                "salience": brand_sal
            })
    return pd.DataFrame(cep_results), msa_df

def simulate_unique_reach(X, elig, wts, current_sal, uplifts, n_sims=800):
    rng = np.random.default_rng(7)
    target_sal = np.minimum(current_sal + (uplifts / 100.0), 0.95)
    
    # Calculate flip probability per person
    denom = (1.0 - current_sal)
    need = (target_sal - current_sal)
    flip_prob = np.where(denom > 0, need / denom, 0.0)
    
    n, k = X.shape
    results = []
    for _ in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        # Flip if Aware, not currently associating, and random check passes
        flips = elig & (X_sim == 0) & (U < flip_prob)
        X_sim[flips] = 1
        results.append(np.sum((X_sim.max(axis=1) > 0) * wts) / np.sum(wts))
    
    return float(np.mean(results)), target_sal

# ============================================================
# 3. STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="TFM MSA Simulator")
    st.title("TFM Mental Availability Simulator (MSA Filtered)")

    # Load Data
    try:
        df_raw = pd.read_csv(RAW_DATA_FILE, low_memory=False)
    except FileNotFoundError:
        st.error(f"Could not find {RAW_DATA_FILE}. Please ensure it is in the script folder.")
        st.stop()

    # Get MSA specific baselines
    cep_df, msa_raw = get_msa_baselines(df_raw)
    wts = msa_raw[WEIGHT_COL].to_numpy()
    
    # Setup Simulation Matrices
    n, k = len(msa_raw), len(cep_df)
    X = np.zeros((n, k), dtype=int)
    elig = np.zeros((n, k), dtype=bool)
    aware_mask = (msa_raw[AWARE_COL] == 1).to_numpy()
    
    for j, row in cep_df.iterrows():
        X[:, j] = (msa_raw[f"RS8_{int(row['id'])}_1NET"] == 1).astype(int).to_numpy()
        elig[:, j] = aware_mask

    # Sidebar Sliders
    st.sidebar.header("Salience Uplift (MSA Aware %)")
    st.sidebar.info("Baselines below are filtered for TFM MSAs and Brand Awareness.")
    uplifts = []
    for idx, row in cep_df.sort_values("prevalence", ascending=False).iterrows():
        val = st.sidebar.slider(
            f"{row['label']} (Baseline: {row['salience']:.1%})",
            0, 30, 0, key=f"s_{row['id']}"
        )
        uplifts.append(val)

    # Run Simulation
    current_reach = np.sum((X.max(axis=1) > 0) * wts) / np.sum(wts)
    scenario_reach, target_sal = simulate_unique_reach(X, elig, wts, cep_df['salience'].values, np.array(uplifts))

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Mental Penetration", f"{current_reach:.1%}")
    c2.metric("Scenario Reach", f"{scenario_reach:.1%}", f"{(scenario_reach - current_reach):.1%}")
    c3.metric("New Households Gained", f"{(scenario_reach - current_reach) * HOUSEHOLD_BASE_TFM_STATES:,.0f}")

    # Results Table
    results_df = cep_df.copy()
    results_df['Uplift'] = uplifts
    results_df['Scenario Salience'] = target_sal
    st.subheader("MSA CEP Performance")
    st.dataframe(results_df[['label', 'prevalence', 'salience', 'Scenario Salience']], use_container_width=True)

    # Chart
    st.subheader("Salience Movement Trail")
    chart_df = results_df.rename(columns={'salience': 'Current', 'Scenario Salience': 'Scenario', 'prevalence': 'Category Prevalence'})
    
    base = alt.Chart(chart_df).encode(x=alt.X("Category Prevalence", axis=alt.Axis(format='%')))
    trail = base.mark_rule(strokeDash=[4,4], color="gray").encode(y=alt.Y("Current", axis=alt.Axis(format='%')), y2="Scenario")
    points = base.mark_circle(size=200, color="#5b9244").encode(y="Current", tooltip=['label', 'Current', 'Scenario'])
    
    st.altair_chart((trail + points).properties(height=500), use_container_width=True)

if __name__ == "__main__":
    main()
