import re
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# 1. CONFIGURATION
# ============================================================
HOUSEHOLD_BASE_TFM_STATES = 70_132_819 
WEIGHT_COL = "wts"
AWARE_COL = "RS3_1NET" 

CEP_NAME_MAP = {
    1: "Doing my regular weekly grocery shopping",
    2: "Trying to save money on groceries",
    3: "Planning meals / realizing I'm low on supplies",
    4: "Running other errands + groceries",
    5: "Feeling inspired to cook / try something new",
    6: "Looking for healthy or better-quality food options",
    7: "Avoiding crowded stores or long lines",
    8: "Buying ready-to-eat meals to save time",
    9: "Buying groceries online or for pickup",
    10: "Wanting to try new or seasonal products",
    11: "Feeling health-conscious / eat better",
    12: "Short on time but wanting convenient food",
    13: "Hosting guests or preparing for a special meal",
    14: "Shopping for a large family or group",
    15: "Looking for specialty or international foods",
    16: "Choosing environmentally friendly options",
    17: "Needing help or ideas on what to buy",
}

# ============================================================
# 2. CORE ENGINE
# ============================================================

def get_baselines(df):
    # Filter for Aware respondents only for the Salience calculation
    aware_mask = (df[AWARE_COL] == 1)
    w_total = df[WEIGHT_COL].to_numpy()
    w_aware = df.loc[aware_mask, WEIGHT_COL].to_numpy()
    
    results = []
    for i in range(1, 18):
        rs1_col, rs8_col = f"RS1_{i}NET", f"RS8_{i}_1NET"
        if rs1_col in df.columns and rs8_col in df.columns:
            # Category Prevalence (Denominator = Total MSA)
            cat_binary = (df[rs1_col] == 1).astype(int).to_numpy()
            cat_prev = np.sum(cat_binary * w_total) / np.sum(w_total)
            
            # Brand Salience (Denominator = Aware Base)
            # This should produce the 34.1% for CEP 6
            sal_binary_aware = (df.loc[aware_mask, rs8_col] == 1).astype(int).to_numpy()
            brand_sal = np.sum(sal_binary_aware * w_aware) / np.sum(w_aware) if len(w_aware) > 0 else 0
            
            results.append({
                "id": i, 
                "label": CEP_NAME_MAP.get(i, f"CEP {i}"), 
                "prevalence": float(cat_prev), 
                "salience": float(brand_sal)
            })
    return pd.DataFrame(results)

def simulate_unique_reach(X, elig, wts, current_sal, uplifts, n_sims=800):
    rng = np.random.default_rng(7)
    target_sal = np.minimum(current_sal + (uplifts / 100.0), 0.95)
    
    # Flip Probability only applies to those who are AWARE but NOT CURRENTLY associating
    denom = (1.0 - current_sal)
    need = (target_sal - current_sal)
    flip_prob = np.where(denom > 0, need / denom, 0.0)
    
    n, k = X.shape
    sim_mpens = []
    for _ in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        # Logic: Eligible (Aware) AND not yet associated AND roll < probability
        flips = elig & (X_sim == 0) & (U < flip_prob)
        X_sim[flips] = 1
        # Calculate Unique Reach: (At least one 1 in the row)
        unique_reach = (X_sim.max(axis=1) > 0).astype(int)
        sim_mpens.append(np.sum(unique_reach * wts) / np.sum(wts))
        
    return float(np.mean(sim_mpens)), target_sal

# ============================================================
# 3. INTERFACE
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="TFM Aligned Simulator")
    st.title("TFM Mental Availability Simulator")
    st.markdown("### Dashboard Alignment: Awareness-Based Salience")

    # Dynamic File Loading
    df = None
    files = [f for f in os.listdir('.') if f.endswith('.csv') and 'TFM' in f]
    
    if files:
        try:
            # Use utf-8-sig for Mac Excel compatibility
            df = pd.read_csv(files[0], encoding='utf-8-sig', low_memory=False)
            st.sidebar.success(f"Loaded: {files[0]}")
        except:
            df = pd.read_csv(files[0], encoding='latin1', low_memory=False)

    if df is None:
        st.error("No CSV found. Please place 'TFM_RAW_MSA.csv' in the folder.")
        st.stop()

    # Clean data (Fixes ValueError)
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors='coerce').fillna(0)
    df[AWARE_COL] = pd.to_numeric(df[AWARE_COL], errors='coerce').fillna(0)

    # Calculate Baselines
    cep_df = get_baselines(df)
    w_total = df[WEIGHT_COL].to_numpy()
    
    # Prep Matrices
    n, k = len(df), len(cep_df)
    X = np.zeros((n, k), dtype=int)
    elig = np.zeros((n, k), dtype=bool)
    aware_mask = (df[AWARE_COL] == 1).to_numpy()
    
    for j, (idx, row) in enumerate(cep_df.iterrows()):
        col_name = f"RS8_{int(row['id'])}_1NET"
        X[:, j] = (df[col_name] == 1).astype(int).to_numpy()
        elig[:, j] = aware_mask

    # Sidebar
    st.sidebar.header("Salience Uplift (Aware Base %)")
    uplifts = []
    # Display in order of Category Prevalence
    for idx, row in cep_df.sort_values("prevalence", ascending=False).iterrows():
        val = st.sidebar.slider(f"{row['label']} (Base: {row['salience']:.1%})", 0, 40, 0, key=f"s_{row['id']}")
        uplifts.append((row['id'], val))
    
    # Run Simulation
    ordered_uplifts = np.array([u[1] for u in sorted(uplifts, key=lambda x: x[0])])
    current_reach = np.sum((X.max(axis=1) > 0) * w_total) / np.sum(w_total)
    scenario_reach, target_sal = simulate_unique_reach(X, elig, w_total, cep_df['salience'].values, ordered_uplifts)

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Market Penetration", f"{current_reach:.1%}")
    c2.metric("Scenario Reach", f"{scenario_reach:.1%}", f"{(scenario_reach - current_reach):+.1%}")
    c3.metric("New Households Gained", f"{(scenario_reach - current_reach) * HOUSEHOLD_BASE_TFM_STATES:,.0f}")

    # Trail Chart
    st.subheader("Bubble Matrix: Growth Trail")
    chart_df = cep_df.copy()
    chart_df['Scenario'] = target_sal
    chart_df = chart_df.rename(columns={'salience': 'Current', 'prevalence': 'Prevalence'})

    base = alt.Chart(chart_df).encode(x=alt.X("Prevalence", axis=alt.Axis(format='%'), title="Category Prevalence"))
    trail = base.mark_rule(strokeDash=[4,4], color="gray", opacity=0.4).encode(
        y=alt.Y("Current", axis=alt.Axis(format='%'), title="Salience (Aware Base)"),
        y2="Scenario"
    )
    points = base.mark_circle(size=350, color="#5b9244", stroke="white", strokeWidth=1).encode(
        y="Current", tooltip=['label', alt.Tooltip('Current', format='.1%'), alt.Tooltip('Scenario', format='.1%')]
    )
    st.altair_chart((trail + points).properties(height=500), use_container_width=True)

    # Data Table
    st.subheader("CEP Performance Detail")
    table_df = chart_df[['label', 'Prevalence', 'Current', 'Scenario']].copy()
    for col in ['Prevalence', 'Current', 'Scenario']:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.1%}")
    st.dataframe(table_df, use_container_width=True)

if __name__ == "__main__":
    main()
