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
# Global Scaling Constant
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
    """Calculates weighted salience ONLY for those Aware of TFM."""
    aware_mask = (df[AWARE_COL] == 1)
    w_total = df[WEIGHT_COL].to_numpy()
    w_aware = df.loc[aware_mask, WEIGHT_COL].to_numpy()
    
    results = []
    for i in range(1, 18):
        rs1_col, rs8_col = f"RS1_{i}NET", f"RS8_{i}_1NET"
        if rs1_col in df.columns and rs8_col in df.columns:
            # Category Prevalence (Total Market in MSA)
            cat_prev = np.sum((df[rs1_col] == 1) * w_total) / np.sum(w_total)
            # Brand Salience (Among Aware in MSA)
            brand_sal = np.sum((df.loc[aware_mask, rs8_col] == 1) * w_aware) / np.sum(w_aware)
            
            results.append({"id": i, "label": CEP_NAME_MAP.get(i, f"CEP {i}"), 
                            "prevalence": cat_prev, "salience": brand_sal})
    return pd.DataFrame(results)

def simulate_reach(X, elig, wts, current_sal, uplifts, n_sims=800):
    rng = np.random.default_rng(7)
    target_sal = np.minimum(current_sal + (uplifts / 100.0), 0.95)
    
    denom = (1.0 - current_sal)
    need = (target_sal - current_sal)
    flip_prob = np.where(denom > 0, need / denom, 0.0)
    
    n, k = X.shape
    sim_mpen = []
    for _ in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        flips = elig & (X_sim == 0) & (U < flip_prob)
        X_sim[flips] = 1
        sim_mpen.append(np.sum((X_sim.max(axis=1) > 0) * wts) / np.sum(wts))
    
    return float(np.mean(sim_mpen)), target_sal

# ============================================================
# 3. STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="TFM MSA Simulator")
    st.title("TFM Mental Availability & TAM Simulator")

    # --- AUTO-DETECT FILE LOGIC ---
    df = None
    # Look for any CSV file in the current directory that fits the TFM pattern
    files = [f for f in os.listdir('.') if f.endswith('.csv') and 'TFM' in f]
    
    if files:
        target_file = files[0] # Take the first matching file
        try:
            df = pd.read_csv(target_file, encoding='latin1', low_memory=False)
            st.sidebar.success(f"Auto-loaded: {target_file}")
        except Exception as e:
            st.sidebar.error(f"Error loading {target_file}: {e}")

    # Fallback to uploader if no file found
    if df is None:
        uploaded_file = st.sidebar.file_uploader("Upload TFM MSA CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='latin1', low_memory=False)
        else:
            st.info("Please upload the MSA CSV or place it in the app directory.")
            st.stop()

    # --- PROCESS DATA ---
    if AWARE_COL not in df.columns:
        st.error(f"Could not find Awareness column ({AWARE_COL}). Check column names.")
        st.stop()

    cep_df = get_baselines(df)
    w_total = df[WEIGHT_COL].to_numpy()
    
    n, k = len(df), len(cep_df)
    X, elig = np.zeros((n, k), dtype=int), np.zeros((n, k), dtype=bool)
    aware_mask = (df[AWARE_COL] == 1).to_numpy()
    for j, (idx, row) in enumerate(cep_df.iterrows()):
        X[:, j] = (df[f"RS8_{int(row['id'])}_1NET"] == 1).astype(int).to_numpy()
        elig[:, j] = aware_mask

    # Sidebar
    st.sidebar.header("Salience Uplift (Aware Base %)")
    uplifts = []
    # Order by prevalence for the UI
    for idx, row in cep_df.sort_values("prevalence", ascending=False).iterrows():
        val = st.sidebar.slider(f"{row['label']} (Base: {row['salience']:.1%})", 0, 40, 0, key=f"s_{row['id']}")
        uplifts.append((row['id'], val))
    
    # Run Math
    u_array = np.array([u[1] for u in sorted(uplifts, key=lambda x: x[0])])
    current_reach = np.sum((X.max(axis=1) > 0) * w_total) / np.sum(w_total)
    scenario_reach, target_sal = simulate_reach(X, elig, w_total, cep_df['salience'].values, u_array)

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Current Market Penetration", f"{current_reach:.1%}")
    k2.metric("Scenario Reach", f"{scenario_reach:.1%}", f"{(scenario_reach - current_reach):+.1%}")
    k3.metric("New Households Gained", f"{(scenario_reach - current_reach) * HOUSEHOLD_BASE_TFM_STATES:,.0f}")

    # Charts & Tables
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

    st.subheader("CEP Performance Detail")
    st.dataframe(chart_df[['label', 'Prevalence', 'Current', 'Scenario']].style.format(precision=1, formatter="{:.1%}"), use_container_width=True)

if __name__ == "__main__":
    main()
