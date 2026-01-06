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
    This simulator models **Unique Household Reach**. Growing associations in overlapping 
    CEPs will yield lower incremental gains than growing distinct entry points.
""")

# -----------------------------
# 2. Data Loading & Cleaning
# -----------------------------
@st.cache_data
def load_and_clean_data():
    try:
        # Fixed: Removed low_memory=False which conflicted with engine='python'
        raw = pd.read_csv(
            RAW_DATA_PATH, 
            engine='python', 
            on_bad_lines='skip'
        )
        # Clean column names (removes hidden spaces or newline chars like 'wts\n')
        raw.columns = raw.columns.str.strip()
        
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame(), []

    # Define the 17 CEP columns for TFM (Brand 1)
    cep_cols = [f"RS8_{i}_1NET" for i in range(1, 18)]
    
    # Ensure columns exist and convert to binary (1=Selected, 0=Not)
    target_cols = cep_cols + ["RS8mpen_1", "wts"]
    for col in target_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0)
            # For survey data where 1=Yes, 2=No, we flip 2 to 0
            if col != "wts":
                raw[col] = np.where(raw[col] == 1, 1, 0)
    
    # Drop rows where weights are missing or zero
    raw = raw[raw["wts"] > 0].copy()
    
    return raw, [c for c in cep_cols if c in raw.columns]

raw, cep_cols = load_and_clean_data()

# Check if data loaded correctly
if raw.empty:
    st.stop()

# CEP Names mapping
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
    uplifts[col] = st.sidebar.slider(f"{name}", 0.0, 0.20, 0.0, 0.01)

# -----------------------------
# 4. Simulation Engine (Deduplicated Math)
# -----------------------------
weights_sum = raw["wts"].sum()

# A. Calculate Baseline Unique Reach
# Mental Penetration is the weighted average of respondents with 1+ associations
baseline_reach = (raw["RS8mpen_1"] * raw["wts"]).sum() / weights_sum
baseline_hhs = baseline_reach * TFM_HOUSEHOLDS

# B. Calculate Expected Scenario Reach
# Start with probability that a respondent is currently NOT reached (1.0 if RS8mpen_1 is 0)
prob_not_reached_scenario = (1.0 - raw["RS8mpen_1"]).astype(float)

for col in cep_cols:
    delta = uplifts[col]
    if delta > 0:
        p_i = (raw[col] * raw["wts"]).sum() / weights_sum
        
        if p_i < 0.99: # Avoid division by zero
            # Probability of flipping a '0' to a '1'
            q_i = delta / (1.0 - p_i)
            
            # Logic: P(Remain 0) = P(Currently 0) * (1 - prob of flipping)
            # We apply this only to respondents who currently do NOT associate with this CEP
            raw_is_zero = (raw[col] == 0)
            prob_not_reached_scenario *= np.where(raw_is_zero, (1.0 - q_i), 1.0)

# Final Unique Reach = 1 - (Weighted prob of not being reached by any point)
expected_reach_scenario = 1.0 - ((prob_not_reached_scenario * raw["wts"]).sum() / weights_sum)
scenario_hhs = expected_reach_scenario * TFM_HOUSEHOLDS
delta_hhs = scenario_hhs - baseline_hhs

# -----------------------------
# 5. Dashboard Display
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Baseline Unique Reach", f"{baseline_reach:.1%}")
with col2:
    st.metric("Scenario Unique Reach", f"{expected_reach_scenario:.1%}", f"{expected_reach_scenario - baseline_reach:+.1%}")
with col3:
    st.metric("New Households Gained", f"{delta_hhs:,.0f}")

# Diagnostic Table
df_diag = pd.DataFrame([
    {
        "CEP": CEP_NAMES.get(col, col),
        "Current Salience": (raw[col] * raw["wts"]).sum() / weights_sum,
        "Scenario Salience": min(((raw[col] * raw["wts"]).sum() / weights_sum) + uplifts[col], 1.0)
    } for col in cep_cols
])

st.subheader("Salience Growth Diagnostic")
chart = alt.Chart(df_diag).mark_bar(color="#6ab04c").encode(
    x=alt.X('Current Salience:Q', axis=alt.Axis(format='%')),
    y=alt.Y('CEP:N', sort='-x')
)
tick = alt.Chart(df_diag).mark_tick(color="red", size=20).encode(
    x='Scenario Salience:Q',
    y='CEP:N'
)
st.altair_chart(chart + tick, use_container_width=True)
