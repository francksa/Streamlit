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
    """Handles percentages, decimals, and string formatting."""
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip().replace(',', '')
    has_pct = '%' in s
    s = s.replace('%', '')
    try:
        f = float(s)
        # Handle cases where 48% is exported as 48 (integers) vs 0.48
        if has_pct or f > 1.0:
            return f / 100.0
        return f
    except ValueError:
        return 0.0

def load_and_align_data():
    if not Path(DASHBOARD_EXPORT).exists() or not Path(RAW_DATA_FILE).exists():
        st.error(f"Missing Files! App needs `{DASHBOARD_EXPORT}` and `{RAW_DATA_FILE}`.")
        st.stop()

    # Load with flexible encoding for Mac/Windows
    try:
        dash_df = pd.read_csv(DASHBOARD_EXPORT, encoding='utf-8-sig')
        raw_df = pd.read_csv(RAW_DATA_FILE, encoding='utf-8-sig', low_memory=False)
    except:
        dash_df = pd.read_csv(DASHBOARD_EXPORT, encoding='latin1')
        raw_df = pd.read_csv(RAW_DATA_FILE, encoding='latin1', low_memory=False)

    raw_df[WEIGHT_COL] = pd.to_numeric(raw_df[WEIGHT_COL], errors='coerce').fillna(0.0)

    # Split dashboard into Overall vs Brand sections
    # Your export has overall prevalence first, then brand salience later
    text_col = dash_df.iloc[:, 0].astype(str)
    brand_start = text_col.str.contains("Your Brand's CEP Salience", case=False, na=False).idxmax()
    
    cat_section = dash_df.iloc[:brand_start]
    brand_section = dash_df.iloc[brand_start:]

    baselines = []
    for cid, label in CEP_NAME_MAP.items():
        # Get Prevalence from the first half of the file
        cat_match = cat_section[cat_section.iloc[:, 0].str.contains(label, na=False, case=False)]
        # Get Salience from the second half
        brand_match = brand_section[brand_section.iloc[:, 0].str.contains(label, na=False, case=False)]
        
        if not cat_match.empty and not brand_match.empty:
            baselines.append({
                "id": cid,
                "label": label,
                "prevalence": clean_numeric(cat_match.iloc[0, 1]),
                "salience": clean_numeric(brand_match.iloc[0, 1])
            })
            
    return pd.DataFrame(baselines), raw_df

# ============================================================
# 3. MONTE CARLO ENGINE
# ============================================================

def run_hh_simulation(raw_df, cep_df, uplifts, n_sims=800):
    rng = np.random.default_rng(7)
    wts = raw_df[WEIGHT_COL].to_numpy()
    n, k = len(raw_df), len(cep_df)
    
    # Association Matrix
    X = np.zeros((n, k), dtype=int)
    elig = (raw_df[AWARE_COL] == 1).to_numpy() 
    
    curr_s = cep_df['salience'].values
    target_s = np.minimum(curr_s + (uplifts / 100.0), 0.95)
    
    # Fill baseline from raw respondent data
    for j, cid in enumerate(cep_df['id']):
        X[:, j] = (raw_df[f"RS8_{cid}_1NET"] == 1).astype(int).to_numpy()

    # P(flip) logic
    denom = (1.0 - curr_s)
    need = (target_s - curr_s)
    flip_probs = np.where(denom > 0, need / denom, 0.0)

    curr_mpen = np.sum((X.max(axis=1) > 0) * wts) / np.sum(wts)

    sim_results = []
    for _ in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        # Flip logic: Aware, currently 0, wins random roll
        flips = (X_sim == 0) & (elig[:, None]) & (U < flip_probs)
        X_sim[flips] = 1
        sim_results.append(np.sum((X_sim.max(axis=1) > 0) * wts) / np.sum(wts))
        
    return curr_mpen, float(np.mean(sim_results)), target_s

# ============================================================
# 4. APP UI
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="TFM HH Simulator")
    st.title("TFM Household Growth & Mental Availability Simulator")

    cep_df, raw_df = load_and_align_data()

    # Sidebar Sliders
    st.sidebar.header("Salience Uplift (pts)")
    uplifts_input = []
    # Key fix: sorting by "prevalence" which is now guaranteed to exist in cep_df
    sorted_ceps = cep_df.sort_values("prevalence", ascending=False)
    
    for idx, row in sorted_ceps.iterrows():
        val = st.sidebar.slider(f"{row['label']} (Base: {row['salience']:.1%})", 0, 30, 0, key=f"s_{row['id']}")
        uplifts_input.append((row['id'], val))
    
    # Simulation execution
    u_array = np.array([u[1] for u in sorted(uplifts_input, key=lambda x: x[0])])
    curr_mpen, scen_mpen, target_saliences = run_hh_simulation(raw_df, cep_df, u_array)
    
    # HH Growth Math
    curr_hhs = curr_mpen * HH_UNIVERSE
    scen_hhs = scen_mpen * HH_UNIVERSE
    new_hhs = scen_hhs - curr_hhs

    # Metrics Row
    st.subheader("Simulated Market Impact")
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Households", f"{curr_hhs:,.0f}")
    m2.metric("Scenario Total Households", f"{scen_hhs:,.0f}")
    m3.metric("Incremental HHs Gained", f"{new_hhs:,.0f}", delta_color="normal")

    # Chart
    st.subheader("Mental Availability Growth Matrix")
    chart_df = cep_df.copy()
    chart_df['Scenario'] = target_saliences
    chart_df = chart_df.rename(columns={'salience': 'Current', 'prevalence': 'Prevalence'})
    
    base = alt.Chart(chart_df).encode(x=alt.X("Prevalence", axis=alt.Axis(format='%'), title="Category Prevalence (Purple Bars)"))
    trail = base.mark_rule(strokeDash=[4,4], color="gray").encode(y="Current", y2="Scenario")
    points = base.mark_circle(size=400, color="#5b9244", stroke="white", strokeWidth=1).encode(
        y=alt.Y("Current", axis=alt.Axis(format='%'), title="Salience (Aware Base)"),
        tooltip=['label', alt.Tooltip('Prevalence', format='.1%'), alt.Tooltip('Current', format='.1%'), alt.Tooltip('Scenario', format='.1%')]
    )
    st.altair_chart((trail + points).properties(height=500), use_container_width=True)

    # Detailed Results Table
    st.subheader("CEP Performance Detail")
    table_df = chart_df[['label', 'Prevalence', 'Current', 'Scenario']].copy()
    for col in ['Prevalence', 'Current', 'Scenario']:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.1%}")
    st.dataframe(table_df, use_container_width=True)

if __name__ == "__main__":
    main()
