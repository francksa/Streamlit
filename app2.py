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
HOUSEHOLD_BASE_TFM_STATES = 70_132_819 
MSA_FILTER_COL = "xdemAud1"
MSA_FILTER_VALUE = 1
WEIGHT_COL = "wts"
AWARE_COL = "RS3_1NET" 

CEP_NAME_MAP = {
    1: "Doing my regular weekly grocery shopping",
    2: "Trying to save money on groceries",
    3: "Planning meals / low on supplies",
    4: "Running other errands",
    5: "Inspired to cook / try something new",
    6: "Looking for healthy or better-quality options",
    7: "Avoiding crowded stores or long lines",
    8: "Buying ready-to-eat meals to save time",
    9: "Buying groceries online or for pickup",
    10: "Wanting to try new or seasonal products",
    11: "Feeling health-conscious / eat better",
    12: "Short on time but wanting convenient food",
    13: "Hosting guests / special meal",
    14: "Shopping for a large family / group",
    15: "Looking for specialty / international foods",
    16: "Choosing eco-friendly grocery options",
    17: "Needing help or ideas on what to buy",
}

# ============================================================
# 2. CALCULATION ENGINE
# ============================================================

def get_msa_baselines(df):
    """Calculates salience % ONLY for MSA respondents who are Aware of TFM."""
    # Filter for MSA respondents
    msa_df = df[df[MSA_FILTER_COL] == MSA_FILTER_VALUE].copy()
    
    # Pre-calculate masks and weights to avoid shape errors
    aware_mask = (msa_df[AWARE_COL] == 1)
    # Weights for only the Aware respondents
    w_aware = msa_df.loc[aware_mask, WEIGHT_COL].to_numpy()
    # Weights for the total MSA market
    w_total = msa_df[WEIGHT_COL].to_numpy()
    
    cep_results = []
    for i in range(1, 18):
        rs1_col = f"RS1_{i}NET"
        rs8_col = f"RS8_{i}_1NET"
        
        if rs1_col in msa_df.columns and rs8_col in msa_df.columns:
            # 1. Category Prevalence (Total MSA Market)
            cat_binary = (msa_df[rs1_col] == 1).astype(int).to_numpy()
            cat_prev = np.sum(cat_binary * w_total) / np.sum(w_total)
            
            # 2. Brand Salience (ONLY Among Aware in MSA)
            brand_sal = 0.0
            if aware_mask.sum() > 0:
                # Get associations only for those who are aware
                sal_binary_aware = (msa_df.loc[aware_mask, rs8_col] == 1).astype(int).to_numpy()
                brand_sal = np.sum(sal_binary_aware * w_aware) / np.sum(w_aware)
            
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
    
    # Calculate flip probability per CEP association
    denom = (1.0 - current_sal)
    need = (target_sal - current_sal)
    # Vectorized flip probability calculation
    flip_prob = np.where(denom > 0, need / denom, 0.0)
    
    n, k = X.shape
    sim_mpen = []
    for _ in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        # Flip only if: Aware (elig), currently 0, and random check passes
        flips = elig & (X_sim == 0) & (U < flip_prob)
        X_sim[flips] = 1
        # Reach = any association selected
        unique_reach = (X_sim.max(axis=1) > 0).astype(int)
        sim_mpen.append(np.sum(unique_reach * wts) / np.sum(wts))
    
    return float(np.mean(sim_mpen)), target_sal

# ============================================================
# 3. STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="TFM Simulator")
    st.title("TFM Mental Availability & TAM Simulator")

    # Load Data
    try:
        df_raw = pd.read_csv(RAW_DATA_FILE, low_memory=False)
    except FileNotFoundError:
        st.error(f"Missing File: {RAW_DATA_FILE}")
        st.stop()

    # Get MSA specific baselines
    cep_df, msa_raw = get_msa_baselines(df_raw)
    w_total_msa = msa_raw[WEIGHT_COL].to_numpy()
    
    # Prepare Matrices for Simulation (n_respondents x k_ceps)
    n, k = len(msa_raw), len(cep_df)
    X = np.zeros((n, k), dtype=int)
    elig = np.zeros((n, k), dtype=bool)
    aware_mask = (msa_raw[AWARE_COL] == 1).to_numpy()
    
    for j, (idx, row) in enumerate(cep_df.iterrows()):
        col_name = f"RS8_{int(row['id'])}_1NET"
        X[:, j] = (msa_raw[col_name] == 1).astype(int).to_numpy()
        elig[:, j] = aware_mask

    # Sidebar
    st.sidebar.header("Salience Uplift (Aware Base %)")
    uplifts = []
    # Display sliders sorted by Category Prevalence (Impact order)
    sorted_df = cep_df.sort_values("prevalence", ascending=False)
    for idx, row in sorted_df.iterrows():
        val = st.sidebar.slider(
            f"{row['label']} (Baseline: {row['salience']:.1%})",
            0, 30, 0, key=f"s_{row['id']}"
        )
        # Store uplift in the original order of cep_df for matrix math
        uplifts.append((row['id'], val))
    
    # Re-order uplifts to match cep_df rows
    uplift_array = np.array([u[1] for u in sorted(uplifts, key=lambda x: x[0])])

    # Run Calculations
    current_reach = np.sum((X.max(axis=1) > 0) * w_total_msa) / np.sum(w_total_msa)
    scenario_reach, target_sal = simulate_unique_reach(X, elig, w_total_msa, cep_df['salience'].values, uplift_array)

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Market Mental Penetration", f"{current_reach:.1%}")
    c2.metric("Scenario Reach", f"{scenario_reach:.1%}", f"{(scenario_reach - current_reach):+.1%}")
    c3.metric("New Households Gained", f"{(scenario_reach - current_reach) * HOUSEHOLD_BASE_TFM_STATES:,.0f}")

    # Data Display
    st.subheader("MSA Category Entry Point Performance")
    results_display = cep_df.copy()
    results_display['Target Salience'] = target_sal
    st.dataframe(results_display[['label', 'prevalence', 'salience', 'Target Salience']].style.format({
        'prevalence': '{:.1%}', 'salience': '{:.1%}', 'Target Salience': '{:.1%}'
    }), use_container_width=True)

    # Growth Trail Chart
    st.subheader("Bubble Matrix: Growth Trail")
    chart_df = results_display.copy()
    chart_df = chart_df.rename(columns={'salience': 'Current', 'Target Salience': 'Scenario'})
    
    base = alt.Chart(chart_df).encode(x=alt.X("prevalence", axis=alt.Axis(format='%'), title="Category Prevalence"))
    trail = base.mark_rule(strokeDash=[4,4], color="gray", opacity=0.5).encode(
        y=alt.Y("Current", axis=alt.Axis(format='%'), title="Salience among Aware"),
        y2="Scenario"
    )
    points = base.mark_circle(size=300, color="#5b9244", stroke="white", strokeWidth=1).encode(
        y="Current",
        tooltip=['label', alt.Tooltip('Current', format='.1%'), alt.Tooltip('Scenario', format='.1%')]
    )
    
    st.altair_chart((trail + points).properties(height=500), use_container_width=True)

if __name__ == "__main__":
    main()
