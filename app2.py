import re
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================
HOUSEHOLD_BASE_TFM_STATES = 70_132_819 
WEIGHT_COL = "wts"
AWARE_COL = "RS3_1NET" 

# Use the exact filenames found in your directory
RAW_DATA_FILE = "TFM_RAW_MSA.xlsx - TFM_RAW_MSA.csv"
DASHBOARD_EXPORT = "Category Entry Points (CEPs).xlsx - Category Entry Points (CEPs).csv"

# ============================================================
# 2. ROBUST HELPERS
# ============================================================

def clean_numeric(val):
    """Handles percentages like '33.7%', decimals like 0.33, or strings."""
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip().replace(',', '')
    if not s: return 0.0
    has_pct = '%' in s
    s = s.replace('%', '')
    try:
        f = float(s)
        return f / 100.0 if (has_pct or f > 1.0) else f
    except ValueError:
        return 0.0

def safe_load_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
    except:
        return pd.read_csv(file_path, encoding='latin1', low_memory=False)

# ============================================================
# 3. DATA ALIGNMENT
# ============================================================

def load_and_align_data():
    if not Path(DASHBOARD_EXPORT).exists() or not Path(RAW_DATA_FILE).exists():
        raise FileNotFoundError("Check filenames: TFM_RAW_MSA and Category Entry Points (CEPs) CSVs needed.")

    dash_df = safe_load_csv(DASHBOARD_EXPORT)
    raw_df = safe_load_csv(RAW_DATA_FILE)
    
    # Map for simulation
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

    # Locate the brand section (usually the second time labels appear)
    brand_start_idx = 0
    for i, val in enumerate(dash_df.iloc[:, 0]):
        if "Your Brand's CEP Salience" in str(val):
            brand_start_idx = i
            break
    brand_section = dash_df.iloc[brand_start_idx:]

    baselines = []
    for cid, label in mapping.items():
        match = brand_section[brand_section.iloc[:, 0].str.contains(label, na=False, case=False)]
        if not match.empty:
            # FIX: Cleaned assignment logic to avoid SyntaxError
            sal_val = clean_numeric(match.iloc[0, 1])
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

    denom = (1.0 - current_s)
    need = (target_s - current_s)
    flip_probs = np.where(denom > 0, need / denom, 0.0)

    sim_reach = []
    for _ in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        # respondents only flip if Aware AND not currently associating
        flips = (X_sim == 0) & (elig[:, None]) & (U < flip_probs)
        X_sim[flips] = 1
        sim_reach.append(np.sum((X_sim.max(axis=1) > 0) * wts) / np.sum(wts))
        
    return float(np.mean(sim_reach)), target_s

# ============================================================
# 5. MAIN UI
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="TFM Mental Availability Simulator")
    st.title("TFM Mental Availability & TAM Simulator")

    try:
        cep_df, raw_df = load_and_align_data()
        st.sidebar.success("✅ Dashboard Export & Raw Data Loaded")
    except Exception as e:
        st.error(f"Error initializing data: {e}")
        st.stop()

    # --- Sidebar ---
    st.sidebar.header("Salience Uplift (pts)")
    uplifts = []
    # Display sliders sorted by Prevalence
    for idx, row in cep_df.sort_values("prevalence", ascending=False).iterrows():
        val = st.sidebar.slider(f"{row['label']} (Base: {row['salience']:.1%})", 0, 25, 0, key=f"s_{row['id']}")
        uplifts.append((row['id'], val))
    
    # --- Simulation ---
    u_array = np.array([u[1] for u in sorted(uplifts, key=lambda x: x[0])])
    scenario_mpen, target_saliences = run_simulation(raw_df, cep_df, u_array)
    
    w = pd.to_numeric(raw_df[WEIGHT_COL], errors='coerce').fillna(1.0).to_numpy()
    current_mpen = np.sum(((raw_df.filter(regex='RS8_.*_1NET') == 1).max(axis=1)) * w) / w.sum()

    # --- KPIs ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Market Penetration", f"{current_mpen:.1%}")
    c2.metric("Scenario Reach", f"{scenario_mpen:.1%}", f"{(scenario_mpen - current_mpen):+.1%}")
    c3.metric("New Households Gained", f"{(scenario_mpen - current_mpen) * HOUSEHOLD_BASE_TFM_STATES:,.0f}")

    # --- Bubble Trail Chart ---
    
    st.subheader("Mental Availability Growth Trail")
    chart_df = cep_df.copy()
    chart_df['Scenario'] = target_saliences
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

    # --- Data Table ---
    st.subheader("Detailed Results")
    res_df = chart_df[['label', 'Prevalence', 'Current', 'Scenario']].copy()
    for col in ['Prevalence', 'Current', 'Scenario']:
        res_df[col] = res_df[col].apply(lambda x: f"{x:.1%}")
    st.dataframe(res_df, use_container_width=True)

if __name__ == "__main__":
    main()
