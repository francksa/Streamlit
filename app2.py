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
                # Convert 1 (Yes) to 1, everything else (2/NA) to 0
                raw[col] = np.where(raw[col] == 1, 1, 0)
    
    raw = raw[raw["wts"] > 0].copy()
    return raw, [c for c in cep_cols if c in raw.columns]

raw, cep_cols = load_and_clean_data()

if raw.empty:
    st.stop()

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
# 3. Sidebar - Unweighted Sliders (FIXED)
# -----------------------------
st.sidebar.header("Salience Uplift (Survey %)")
st.sidebar.write("Add percentage points (+Δ) to the **Unweighted** survey results.")

uplifts = {}
for col in cep_cols:
    name = CEP_NAMES.get(col, col)
    current_unweighted = raw[col].mean()
    
    # FIX: Use integers (0-20) for percentage points to avoid sprintf errors.
    # format="%d%%" is the standard way to show a '%' sign in the slider interface.
    delta_pts = st.sidebar.slider(
        f"{name} (Baseline: {current_unweighted:.1%})", 
        min_value=0, 
        max_value=20, 
        value=0, 
        step=1,
        format="%d%%" 
    )
    # Convert the integer points (e.g., 5) back to decimal (0.05) for the math
    uplifts[col] = delta_pts / 100.0

# -----------------------------
# 4. Simulation Engine
# -----------------------------
weights_sum = raw["wts"].sum()
baseline_reach_weighted = (raw["RS8mpen_1"] * raw["wts"]).sum() / weights_sum
baseline_hhs = baseline_reach_weighted * TFM_HOUSEHOLDS

prob_not_reached_scenario = (1.0 - raw["RS8mpen_1"]).astype(float)

for col in cep_cols:
    delta = uplifts[col]
    if delta > 0:
        p_unw = raw[col].mean()
        if p_unw < 0.99:
            q_i = delta / (1.0 - p_unw)
            raw_is_zero = (raw[col] == 0)
            prob_not_reached_scenario *= np.where(raw_is_zero, (1.0 - q_i), 1.0)

expected_reach_weighted_scenario = 1.0 - ((prob_not_reached_scenario * raw["wts"]).sum() / weights_sum)
scenario_hhs = expected_reach_weighted_scenario * TFM_HOUSEHOLDS
delta_hhs = scenario_hhs - baseline_hhs

# -----------------------------
# 5. Dashboard Display
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Market Mental Penetration", f"{baseline_reach_weighted:.1%}")
with col2:
    st.metric("Scenario Reach", f"{expected_reach_weighted_scenario:.1%}", f"{expected_reach_weighted_scenario - baseline_reach_weighted:+.1%}")
with col3:
    st.metric("New Households Gained", f"{delta_hhs:,.0f}")

# Chart Diagnostic
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
