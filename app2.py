#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 18:23:55 2026

@author: franck
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -----------------------------
# 1. Configuration & Assumptions
# -----------------------------
TFM_HOUSEHOLDS = 70_132_819
RAW_DATA_PATH = "https://github.com/francksa/Streamlit/blob/main/2510110_research_science_raw_data.csv"

st.set_page_config(page_title="TFM Mental Availability Simulator", layout="wide")
st.title("TFM Mental Availability & TAM Simulator")
st.markdown("""
    This simulator models **Unique Household Reach**. Growing associations in overlapping 
    CEPs (e.g., Weekly Shopping and Errands) will yield lower incremental gains than growing 
    distinct entry points.
""")

# -----------------------------
# 2. Data Loading & Cleaning
# -----------------------------
@st.cache_data
def load_and_clean_data():
    try:
        # 1. Added engine='python' (more flexible than the C engine)
        # 2. Added on_bad_lines='skip' to bypass the broken rows
        # 3. Added low_memory=False to handle mixed data types in large files
        raw = pd.read_csv(
            RAW_DATA_PATH, 
            engine='python', 
            on_bad_lines='skip', 
            low_memory=False
        )
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame(), []

    # Map RS8_1_1NET through RS8_17_1NET
    cep_cols = [f"RS8_{i}_1NET" for i in range(1, 18)]
    
    # Check if columns actually exist before trying to convert
    available_cep_cols = [c for c in cep_cols if c in raw.columns]
    
    for col in available_cep_cols + ["RS8mpen_1"]:
        if col in raw.columns:
            # Handle potential strings/NaNs in the column before conversion
            raw[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0)
            raw[col] = np.where(raw[col] == 1, 1, 0)
    
    raw = raw.dropna(subset=['wts'])
    return raw, available_cep_cols

raw, cep_cols = load_and_clean_data()

# CEP Names mapping (Based on your provided CEP_DATA logic)
CEP_NAMES = {
    "RS8_1_1NET": "Weekly grocery shopping",
    "RS8_2_1NET": "Trying to save money",
    "RS8_3_1NET": "Planning meals / low on supplies",
    "RS8_4_1NET": "Errands + groceries",
    "RS8_5_1NET": "Inspired to cook",
    "RS8_6_1NET": "Exploring new flavors",
    "RS8_7_1NET": "Healthy / organic foods",
    "RS8_8_1NET": "Quick / convenient meal",
    "RS8_9_1NET": "Special occasion / entertaining",
    "RS8_10_1NET": "Treat / splurge",
    "RS8_11_1NET": "Fresh produce",
    "RS8_12_1NET": "High-quality meat / seafood",
    "RS8_13_1NET": "Bakery / prepared foods",
    "RS8_14_1NET": "Curated / unique selection",
    "RS8_15_1NET": "Helpful / friendly staff",
    "RS8_16_1NET": "Pleasant shopping environment",
    "RS8_17_1NET": "Supporting local"
}

# -----------------------------
# 3. Sidebar - Scenario Inputs
# -----------------------------
st.sidebar.header("Salience Uplift Scenarios")
st.sidebar.write("Apply percentage point (+Δ) increases to TFM salience per CEP.")

uplifts = {}
for col in cep_cols:
    name = CEP_NAMES.get(col, col)
    # Slider provides 0-20 percentage point uplift
    uplifts[col] = st.sidebar.slider(f"{name}", 0.0, 0.20, 0.0, 0.01)

# -----------------------------
# 4. Simulation Engine (Deduplicated Math)
# -----------------------------

# A. Calculate Baseline Unique Reach
weights_sum = raw["wts"].sum()
baseline_reach = (raw["RS8mpen_1"] * raw["wts"]).sum() / weights_sum
baseline_hhs = baseline_reach * TFM_HOUSEHOLDS

# B. Calculate Expected Scenario Reach
# We start with the probability that a respondent is currently NOT reached by TFM
# (1.0 if RS8mpen_1 is 0, else 0.0)
prob_not_reached_scenario = (1.0 - raw["RS8mpen_1"]).astype(float)

for col in cep_cols:
    delta = uplifts[col]
    if delta > 0:
        # p_i = Weighted % of sample selecting TFM for this CEP
        p_i = (raw[col] * raw["wts"]).sum() / weights_sum
        
        # q_i = Probability of flipping a '0' to a '1' to achieve the delta
        # Formula: delta = q_i * (1 - p_i)
        if p_i < 0.99: # Avoid division by zero
            q_i = delta / (1.0 - p_i)
            
            # If the respondent currently has a 0 for this CEP, their 
            # probability of 'remaining unreached' is reduced by (1 - q_i)
            raw_is_zero = (raw[col] == 0)
            prob_not_reached_scenario *= np.where(raw_is_zero, (1.0 - q_i), 1.0)

# Final Deduplicated scenario reach %
# Unique Reach = 1 - (Weighted average probability of not being reached)
expected_reach_scenario = 1.0 - ((prob_not_reached_scenario * raw["wts"]).sum() / weights_sum)
scenario_hhs = expected_reach_scenario * TFM_HOUSEHOLDS
delta_hhs = scenario_hhs - baseline_hhs

# -----------------------------
# 5. Dashboard Display
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Baseline Unique Reach", f"{baseline_reach:.1%}", help="Current % of HHs associating TFM with 1+ CEPs")
with col2:
    st.metric("Scenario Unique Reach", f"{expected_reach_scenario:.1%}", f"{expected_reach_scenario - baseline_reach:+.1%}")
with col3:
    st.metric("New Households Gained", f"{delta_hhs:,.0f}", delta_color="normal")

# Diagnostic Dataframe
diagnostic_data = []
for col in cep_cols:
    p_i = (raw[col] * raw["wts"]).sum() / weights_sum
    diagnostic_data.append({
        "CEP": CEP_NAMES.get(col, col),
        "Current Salience": p_i,
        "Uplift Applied": uplifts[col],
        "Scenario Salience": min(p_i + uplifts[col], 1.0)
    })

df_diag = pd.DataFrame(diagnostic_data)

# Visualizing results
st.subheader("Mental Availability by CEP")
chart = alt.Chart(df_diag).mark_bar().encode(
    x=alt.X('Current Salience:Q', axis=alt.Axis(format='%')),
    y=alt.Y('CEP:N', sort='-x'),
    color=alt.value("#6ab04c"),
    tooltip=['CEP', 'Current Salience', 'Scenario Salience']
).properties(height=500)

# Overlay the scenario salience as a tick or second bar
scenario_chart = alt.Chart(df_diag).mark_tick(color="red", size=20).encode(
    x=alt.X('Scenario Salience:Q'),
    y=alt.Y('CEP:N', sort='-x'),
)

st.altair_chart(chart + scenario_chart, use_container_width=True)

st.write("---")
st.caption(f"Calculations based on total market size of {TFM_HOUSEHOLDS:,} households[cite: 1].")
