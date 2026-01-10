import re
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# ============================================================
# 1. FILE CONFIGURATION (Updated to match your files)
# ============================================================
RAW_DATA_FILE = "TFM_RAW_MSA.csv"
DASHBOARD_EXPORT = "Category Entry Points (CEPs).csv"

HOUSEHOLD_BASE_TFM_STATES = 70_132_819 
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
# 2. ROBUST HELPERS
# ============================================================

def clean_numeric(val):
    """Handles strings like '33.7%', '0.33', or ',600' safely."""
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    
    s = str(val).strip().replace(',', '')
    if not s or s.lower() == 'nan': return 0.0
    
    has_pct = '%' in s
    s = s.replace('%', '')
    
    try:
        f = float(s)
        # If it's a whole number > 1 and had a % sign (e.g. '34%'), or just '34'
        # we treat it as a percentage. Dashboards often export '34' or '34%'
        if has_pct or f > 1.0:
            return f / 100.0
        return f
    except ValueError:
        return 0.0

def safe_load_csv(file_path):
    """Mac-proof CSV loader."""
    try:
        return pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
    except:
        return pd.read_csv(file_path, encoding='latin1', low_memory=False)

# ============================================================
# 3. DATA LOADING & ALIGNMENT
# ============================================================

def load_and_align_data():
    if not Path(DASHBOARD_EXPORT).exists() or not Path(RAW_DATA_FILE).exists():
        st.error(f"Missing files in directory: Ensure `{DASHBOARD_EXPORT}` and `{RAW_DATA_FILE}` are present.")
        st.stop()

    dash_df = safe_load_csv(DASHBOARD_EXPORT)
    raw_df = safe_load_csv(RAW_DATA_FILE)
    
    # Locate the section: "Your Brand's CEP Salience"
    brand_start_idx = 0
    for i, val in enumerate(dash_df.iloc[:, 0]):
        if "Your Brand's CEP Salience" in str(val):
            brand_start_idx = i
            break
    
    brand_section = dash_df.iloc[brand_start_idx:]

    baselines = []
    for cid, label in CEP_NAME_MAP.items():
        # Match labels in the dashboard file to get the starting percentages
        match = brand_section[brand_section.iloc[:, 0].str.contains(label, na=False, case=False)]
        if not match.empty:
            sal_val = clean_numeric(match.iloc[0, 1])
            # Prevalence is derived from Raw Data for MSA Universe consistency
            cat_prev = (raw_df[f"RS1_{cid}NET"] == 1).mean() 
            
            baselines.append({
                "id": cid, 
                "label": label, 
                "salience": sal_val, 
                "prevalence": cat_prev
            })
            
    return pd.DataFrame(baselines), raw_df

# ============================================================
# 4. SIMULATION ENGINE
# ============================================================

def run_simulation(raw_df, cep_df, uplifts, n_sims=800):
    rng = np.random.default_rng(7)
    wts = pd.to_numeric(raw_df[WEIGHT_COL], errors='coerce').fillna(1.0).to_numpy()
    
    n, k = len(raw_df), len(cep_df)
    X = np.zeros((n, k), dtype=int)
    elig = (raw_df[AWARE_COL] == 1).to_numpy()
    
    current_s = cep_df['salience'].values
    target_s = np.minimum(current_s + (uplifts / 100.0), 0.95)
    
    for j, cid in enumerate(cep_df['id']):
        X[:, j] = (raw_df[f"RS8_{cid}_1NET"] == 1).astype(int).to_numpy()

    # Probability to 'flip' based on the gap to the target
    denom = (1.0 - current_s)
    need = (target_s - current_s)
    flip_probs = np.where(denom > 0, need / denom, 0.0)

    sim_reach = []
    for _ in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        # Flip respondents who are Aware but don't currently associate TFM with that CEP
        flips = (X_sim == 0) & (elig[:, None]) & (U < flip_probs)
        X_sim[flips] = 1
        # Reach = respondent has at least one brand association
        sim_reach.append(np.sum((X_sim.max(axis=1) > 0) * wts) / np.sum(wts))
        
    return float(np.mean(sim_reach)), target_s

# ============================================================
# 5. MAIN APP
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="TFM Mental Availability Simulator")
    st.title("TFM Mental Availability & TAM Simulator")
    st.info("Start points aligned to Dashboard Export; Growth calculated on Respondent Raw Data.")

    # Load Data
    try:
        cep_df, raw_df = load_and_align_data()
        st.sidebar.success("✅ Files Loaded and Aligned")
    except Exception as e:
        st.error(f"Error initializing data: {e}")
        st.stop()

    # --- Sidebar ---
    st.sidebar.header("Salience Uplift (pts)")
    uplifts = []
    # Display sliders sorted by Category Prevalence (Impact Order)
    for idx, row in cep_df.sort_values("prevalence", ascending=False).iterrows():
        val = st.sidebar.slider(f"{row['label']} (Base: {row['salience']:.1%})", 0, 25, 0, key=f"s_{row['id']}")
        uplifts.append((row['id'], val))
    
    # --- Execute Simulation ---
    u_array = np.array([u[1] for u in sorted(uplifts, key=lambda x: x[0])])
    scenario_mpen, target_saliences = run_simulation(raw_df, cep_df, u_array)
    
    # Baseline Mental Penetration (Weighted)
    w = pd.to_numeric(raw_df[WEIGHT_COL], errors='coerce').fillna(1.0).to_numpy()
    current_mpen = np.sum(((raw_df.filter(regex='RS8_.*_1NET') == 1).max(axis=1)) * w) / w.sum()

    # --- KPI Dashboard ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Market Penetration", f"{current_mpen:.1%}")
    c2.metric("Scenario Reach", f"{scenario_mpen:.1%}", f"{(scenario_mpen - current_mpen):+.1%}")
    c3.metric("New Households Gained", f"{(scenario_mpen - current_mpen) * HOUSEHOLD_BASE_TFM_STATES:,.0f}")

    # --- Bubble Trail Chart ---
    st.subheader("Mental Availability Growth Matrix")
    chart_df = cep_df.copy()
    chart_df['Scenario'] = target_saliences
    chart_df = chart_df.rename(columns={'salience': 'Current', 'prevalence': 'Prevalence'})

    base = alt.Chart(chart_df).encode(x=alt.X("Prevalence", axis=alt.Axis(format='%'), title="Category Prevalence"))
    trail = base.mark_rule(strokeDash=[4,4], color="gray", opacity=0.4).encode(
        y=alt.Y("Current", axis=alt.Axis(format='%'), title="Salience (Among Aware Respondents)"),
        y2="Scenario"
    )
    points = base.mark_circle(size=350, color="#5b9244", stroke="white", strokeWidth=1).encode(
        y="Current", tooltip=['label', alt.Tooltip('Current', format='.1%'), alt.Tooltip('Scenario', format='.1%')]
    )
    st.altair_chart((trail + points).properties(height=500), use_container_width=True)

    # --- Data Table ---
    st.subheader("Detailed Scenario Results")
    res_df = chart_df[['label', 'Prevalence', 'Current', 'Scenario']].copy()
    for col in ['Prevalence', 'Current', 'Scenario']:
        res_df[col] = res_df[col].apply(lambda x: f"{x:.1%}")
    st.dataframe(res_df, use_container_width=True)

if __name__ == "__main__":
    main()
