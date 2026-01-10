import re
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# ============================================================
# 1. FILE & UNIVERSE CONFIGURATION
# ============================================================
RAW_DATA_FILE = "TFM_RAW_MSA.csv"
DASHBOARD_EXPORT = "Category Entry Points (CEPs).csv"

# The Total HH Universe for TFM footprint states
HH_UNIVERSE = 70_132_819 

WEIGHT_COL = "wts"
AWARE_COL = "RS3_1NET" 

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
# 2. DATA UTILITIES
# ============================================================

def clean_numeric(val):
    """Handles percentages (34%), decimals (0.34), and strings safely."""
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip().replace(',', '')
    has_pct = '%' in s
    s = s.replace('%', '')
    try:
        f = float(s)
        return f / 100.0 if (has_pct or f > 1.0) else f
    except ValueError:
        return 0.0

def safe_load():
    if not Path(DASHBOARD_EXPORT).exists() or not Path(RAW_DATA_FILE).exists():
        st.error("Missing files! Please ensure both CSVs are in the directory.")
        st.stop()
    
    # Load Dashboard Values (Truths)
    dash_df = pd.read_csv(DASHBOARD_EXPORT, encoding='utf-8-sig')
    brand_start = 0
    for i, val in enumerate(dash_df.iloc[:, 0]):
        if "Your Brand's CEP Salience" in str(val):
            brand_start = i
            break
    brand_section = dash_df.iloc[brand_start:]

    # Load Raw Respondent Data
    raw_df = pd.read_csv(RAW_DATA_FILE, encoding='utf-8-sig', low_memory=False)
    raw_df[WEIGHT_COL] = pd.to_numeric(raw_df[WEIGHT_COL], errors='coerce').fillna(0.0)

    baselines = []
    for cid, label in CEP_NAME_MAP.items():
        match = brand_section[brand_section.iloc[:, 0].str.contains(label, na=False, case=False)]
        if not match.empty:
            sal_val = clean_numeric(match.iloc[0, 1])
            # Prevalence = % of total MSA respondents who do this behavior
            cat_prev = (raw_df[f"RS1_{cid}NET"] == 1).mean() 
            baselines.append({"id": cid, "label": label, "salience": sal_val, "prevalence": cat_prev})
            
    return pd.DataFrame(baselines), raw_df

# ============================================================
# 3. HOUSEHOLD IMPACT ENGINE
# ============================================================

def run_hh_simulation(raw_df, cep_df, uplifts, n_sims=800):
    rng = np.random.default_rng(7)
    wts = raw_df[WEIGHT_COL].to_numpy()
    
    n, k = len(raw_df), len(cep_df)
    X = np.zeros((n, k), dtype=int)
    elig = (raw_df[AWARE_COL] == 1).to_numpy() # Only Aware HHs can associate
    
    current_s = cep_df['salience'].values
    target_s = np.minimum(current_s + (uplifts / 100.0), 0.95)
    
    # Fill starting associations from raw data
    for j, cid in enumerate(cep_df['id']):
        X[:, j] = (raw_df[f"RS8_{cid}_1NET"] == 1).astype(int).to_numpy()

    # Probability to "flip" a non-associated HH to associated
    denom = (1.0 - current_s)
    need = (target_s - current_s)
    flip_probs = np.where(denom > 0, need / denom, 0.0)

    # Initial State: Who is currently in the brand's mental network?
    current_reach_bool = (X.max(axis=1) > 0)
    current_mpen = np.sum(current_reach_bool * wts) / np.sum(wts)

    sim_mpens = []
    for _ in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        # Flip logic: respondent is Aware, currently 0 for this CEP, and wins the roll
        flips = (X_sim == 0) & (elig[:, None]) & (U < flip_probs)
        X_sim[flips] = 1
        # Final State: Who is in the network now?
        sim_mpens.append(np.sum((X_sim.max(axis=1) > 0) * wts) / np.sum(wts))
        
    avg_scenario_mpen = float(np.mean(sim_mpens))
    return current_mpen, avg_scenario_mpen, target_s

# ============================================================
# 4. USER INTERFACE
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="TFM HH Growth Simulator")
    st.title("TFM Household Growth & Mental Availability Simulator")
    
    cep_df, raw_df = safe_load()

    # Sidebar
    st.sidebar.header("Salience Strategy (Uplift pts)")
    uplifts = []
    for idx, row in cep_df.sort_values("prevalence", ascending=False).iterrows():
        # Baseline is pulled from DASHBOARD_EXPORT (e.g., 33.7%)
        val = st.sidebar.slider(f"{row['label']} (Base: {row['salience']:.1%})", 0, 30, 0, key=f"s_{row['id']}")
        uplifts.append((row['id'], val))
    
    # Run Simulation
    u_array = np.array([u[1] for u in sorted(uplifts, key=lambda x: x[0])])
    curr_mpen, scen_mpen, target_saliences = run_hh_simulation(raw_df, cep_df, u_array)
    
    # Household Math
    curr_hhs = curr_mpen * HH_UNIVERSE
    scen_hhs = scen_mpen * HH_UNIVERSE
    new_hhs = scen_hhs - curr_hhs

    # Display Metrics
    st.subheader("Simulated Market Impact")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Market Penetration", f"{curr_mpen:.1%}", help="Households with at least one TFM association.")
    col2.metric("Scenario Market Penetration", f"{scen_mpen:.1%}", f"{scen_mpen - curr_mpen:+.1%}")
    col3.metric("Incremental Households Gained", f"{new_hhs:,.0f}", delta_color="normal")

    

    # Bubble Chart
    st.subheader("Growth Matrix: Salience vs Prevalence")
    chart_df = cep_df.copy()
    chart_df['Scenario'] = target_saliences
    chart_df = chart_df.rename(columns={'salience': 'Current', 'prevalence': 'Prevalence'})
    
    base = alt.Chart(chart_df).encode(x=alt.X("Prevalence", axis=alt.Axis(format='%'), title="Behavior Prevalence (% of Total Market)"))
    trail = base.mark_rule(strokeDash=[4,4], color="gray").encode(y=alt.Y("Current", axis=alt.Axis(format='%')), y2="Scenario")
    points = base.mark_circle(size=400, color="#5b9244", stroke="white", strokeWidth=1).encode(
        y=alt.Y("Current", axis=alt.Axis(format='%'), title="TFM Salience (Aware Base)"),
        tooltip=['label', alt.Tooltip('Current', format='.1%'), alt.Tooltip('Scenario', format='.1%')]
    )
    st.altair_chart((trail + points).properties(height=500), use_container_width=True)

    # Detailed Table
    st.subheader("CEP Strategic Detail")
    table_df = chart_df[['label', 'Prevalence', 'Current', 'Scenario']].copy()
    for col in ['Prevalence', 'Current', 'Scenario']:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.1%}")
    st.dataframe(table_df, use_container_width=True)

if __name__ == "__main__":
    main()
