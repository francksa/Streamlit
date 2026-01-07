import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -----------------------------
# 1. Configuration & Assumptions
# -----------------------------
TFM_HOUSEHOLDS = 70_132_819
RAW_DATA_PATH = "2510110_research_science_raw_data.csv"

st.set_page_config(page_title="TFM Mental Availability Simulator", layout="wide")
st.title("TFM Mental Availability & TAM Simulator")
st.markdown("""
    **Survey-to-Market Model**: Inputs and Salience bars are **Unweighted** to match your survey tracker. 
    Household Gains are **Weighted** to accurately project to the 70M+ market TAM.
""")

# -----------------------------
# 2. Data Loading & Cleaning
# -----------------------------
@st.cache_data
def load_and_clean_data():
    try:
        raw = pd.read_csv(RAW_DATA_PATH, engine='python', on_bad_lines='skip')
        raw.columns = raw.columns.str.strip()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame(), []

    cep_cols = [f"RS8_{i}_1NET" for i in range(1, 18)]
    target_cols = cep_cols + ["RS8mpen_1", "wts"]
    
    for col in target_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0)
            if col != "wts":
                # Convert 1 (Yes) / 2 (No) to 1 / 0
                raw[col] = np.where(raw[col] == 1, 1, 0)
    
    # Filter for valid data
    raw = raw[raw["wts"] > 0].copy()
    return raw, [c for c in cep_cols if c in raw.columns]

raw, cep_cols = load_and_clean_data()

# Stop if data failed to load
if raw.empty:
    st.warning("Data could not be loaded. Please check the CSV file.")
    st.stop()

# CEP Names mapping
CEP_NAMES = {
    "RS8_1_1NET": "Weekly grocery shopping", "RS8_2_1NET": "Trying to save money",
    "RS8_3_1NET": "Planning meals / supplies", "RS8_4_1NET": "Errands + groceries",
    "RS8_5_1NET": "Inspired to cook", "RS8_6_1NET": "Exploring new flavors",
    "RS8_7_1NET": "Healthy / organic foods", "RS8_8_1NET": "Quick / convenient meal",
    "RS8_9_1NET": "Special occasion", "RS8_10_1NET": "Treat / splurge",
    "RS8_11_1NET": "Fresh produce", "RS8_12_1NET": "High-quality meat/seafood",
    "RS8_13_1NET": "Bakery / prepared foods", "RS8_14_1NET": "Curated selection",
    "RS8_15_1NET": "Helpful staff", "RS8_16_1NET": "Pleasant environment",
    "RS8_17_1NET": "Supporting local"
}

# -----------------------------
# 3. Sidebar - Unweighted Sliders
# -----------------------------
st.sidebar.header("Salience Uplift (Survey %)")
st.sidebar.write("Add percentage points (+Δ) to the **Unweighted** survey results.")

uplifts = {}
for col in cep_cols:
    name = CEP_NAMES.get(col, col)
    # Baseline unweighted salience for the slider label
    current_unweighted = raw[col].mean()
    uplifts[col] = st.sidebar.slider(
        f"{name} (Current: {current_unweighted:.1%})", 
        0.0, 0.20, 0.0, 0.01, 
        format="+.0%"
    )

# -----------------------------
# 4. Simulation Engine (Deduplicated & Weighted)
# -----------------------------

# Step A: Baseline (Weighted Reach for the TAM)
weights_sum = raw["wts"].sum()
baseline_reach_weighted = (raw["RS8mpen_1"] * raw["wts"]).sum() / weights_sum
baseline_hhs = baseline_reach_weighted * TFM_HOUSEHOLDS

# Step B: Scenario Probability Modeling
# Start with 'Prob that respondent is currently NOT reached'
# This is 1.0 if RS8mpen_1 is 0, and 0.0 if they are already associated with TFM
prob_not_reached_scenario = (1.0 - raw["RS8mpen_1"]).astype(float)

for col in cep_cols:
    delta = uplifts[col]
    if delta > 0:
        # Calculate UNWEIGHTED baseline for the 'intuitive' flip math
        p_unweighted = raw[col].mean()
        
        if p_unweighted < 0.99:
            # q_i = Probability of flipping a '0' to a '1' in the survey sample
            q_i = delta / (1.0 - p_unweighted)
            
            # Apply only to people who are currently '0' for this CEP
            # This correctly models the 'Incremental Reach' logic
            raw_is_zero = (raw[col] == 0)
            prob_not_reached_scenario *= np.where(raw_is_zero, (1.0 - q_i), 1.0)

# Step C: Final Result (Weighted Projection)
expected_reach_weighted_scenario = 1.0 - ((prob_not_reached_scenario * raw["wts"]).sum() / weights_sum)
scenario_hhs = expected_reach_weighted_scenario * TFM_HOUSEHOLDS
delta_hhs = scenario_hhs - baseline_hhs

# -----------------------------
# 5. Dashboard Display
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    # We show the Weighted Unique Reach as the true Market baseline
    st.metric("Market Mental Penetration", f"{baseline_reach_weighted:.1%}", help="Weighted % of HHs associating TFM with 1+ CEPs")
with col2:
    st.metric("Scenario Reach", f"{expected_reach_weighted_scenario:.1%}", f"{expected_reach_weighted_scenario - baseline_reach_weighted:+.1%}")
with col3:
    st.metric("New Households Gained", f"{delta_hhs:,.0f}", help="Projected gain across 70,132,819 HHs")

# Diagnostic Table using UNWEIGHTED numbers for client familiarity
diag_list = []
for col in cep_cols:
    curr_unw = raw[col].mean()
    diag_list.append({
        "CEP": CEP_NAMES.get(col, col),
        "Survey Salience (Current)": curr_unw,
        "Survey Salience (Scenario)": min(curr_unw + uplifts[col], 1.0)
    })
df_diag = pd.DataFrame(diag_list)

st.subheader("CEP Salience Growth (Unweighted Survey %)")
base_chart = alt.Chart(df_diag).mark_bar(color="#6ab04c").encode(
    x=alt.X('Survey Salience (Current):Q', axis=alt.Axis(format='%')),
    y=alt.Y('CEP:N', sort='-x')
)
scenario_tick = alt.Chart(df_diag).mark_tick(color="red", size=20, thickness=3).encode(
    x='Survey Salience (Scenario):Q',
    y='CEP:N'
)
st.altair_chart(base_chart + scenario_tick, use_container_width=True)

with st.expander("Technical View: Weighted vs Unweighted Verification"):
    st.dataframe(pd.DataFrame([{
        "CEP": CEP_NAMES.get(col, col),
        "Unweighted (Survey)": f"{raw[col].mean():.1%}",
        "Weighted (Market)": f"{((raw[col] * raw['wts']).sum() / weights_sum):.1%}"
    } for col in cep_cols]))
