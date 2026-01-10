import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# CONFIG
# ============================================================
RAW_DATA_DEFAULT_NAMES = [
    "2510110_research_science_raw_data (1).csv",
    "2510110_research_science_raw_data.csv",
]

HOUSEHOLD_BASE_TFM_STATES = 70_132_819
MSA_FILTER_COL = "xdemAud1"
MSA_FILTER_VALUE = 1
WEIGHT_COL = "wts"
AWARE_COL = "RS3_1NET"  # Brand Awareness Column

DEFAULT_N_SIMS = 800
DEFAULT_CAP = 0.90
APP_DIR = Path(__file__).parent

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

def infer_cep_type(label: str) -> str:
    s = label.lower()
    routine_kw = ["weekly", "regular", "save money", "low on supplies", "planning", "errands"]
    health_kw = ["healthy", "health-conscious", "eat better", "better-quality", "environment"]
    convenience_kw = ["short on time", "convenient", "ready-to-eat", "pickup", "online", "crowded", "long lines", "avoid"]
    inspiration_kw = ["inspired", "try something new", "new or seasonal", "seasonal", "hosting", "special meal", "ideas", "help"]
    specialty_kw = ["specialty", "international", "large family", "group"]
    if any(k in s for k in routine_kw): return "Routine"
    if any(k in s for k in health_kw): return "Health"
    if any(k in s for k in convenience_kw): return "Convenience"
    if any(k in s for k in inspiration_kw): return "Inspiration"
    if any(k in s for k in specialty_kw): return "Specialty"
    return "Other"

# ============================================================
# Utilities
# ============================================================
def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.sum(w))

def load_csv_from_repo_or_upload(label: str, default_names: list[str], required: bool = True):
    for name in default_names:
        p = APP_DIR / name
        if p.exists():
            st.sidebar.success(f"{label}: loaded from repo ({p.name})")
            return pd.read_csv(p, low_memory=False)
    uploaded = st.sidebar.file_uploader(label, type=["csv"])
    if uploaded is None:
        if required:
            st.sidebar.warning(f"Please upload: {label}")
            st.stop()
        return None
    st.sidebar.success(f"{label}: uploaded")
    return pd.read_csv(uploaded, low_memory=False)

def apply_msa_filter(raw: pd.DataFrame, use_msa_only: bool) -> pd.DataFrame:
    if not use_msa_only:
        return raw
    if MSA_FILTER_COL not in raw.columns:
        raise ValueError(f"MSA filter column not found. Expected '{MSA_FILTER_COL}'.")
    return raw[raw[MSA_FILTER_COL] == MSA_FILTER_VALUE].copy()

def discover_ceps(raw: pd.DataFrame):
    cep_idx = []
    for col in raw.columns:
        m = re.match(r"RS1_(\d+)NET$", str(col))
        if m:
            i = int(m.group(1))
            if f"RS8_{i}_1NET" in raw.columns:
                cep_idx.append(i)
    cep_idx = sorted(set(cep_idx))
    return cep_idx

def weighted_category_prevalence(rs1_series: pd.Series, w: np.ndarray) -> float:
    x = (rs1_series == 1).astype(int).to_numpy()
    return wmean(x, w)

# --- UPDATED: Salience is now % of Brand Aware respondents ---
def weighted_brand_salience(rs8_series: pd.Series, aware_series: pd.Series, w: np.ndarray) -> float:
    """
    Calculates weighted % selecting TFM ONLY among those aware of TFM.
    """
    aware_mask = (aware_series == 1).to_numpy()
    if aware_mask.sum() == 0:
        return 0.0
    x = (rs8_series == 1).astype(int).to_numpy()
    # Denominator is sum of weights of AWARE people, Numerator is sum of weights of AWARE people who associated
    return float(np.sum(x[aware_mask] * w[aware_mask]) / np.sum(w[aware_mask]))

def build_X_and_eligibility(raw: pd.DataFrame, aware_series: pd.Series, cep_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    n = len(raw)
    k = len(cep_idx)
    X = np.zeros((n, k), dtype=int)
    elig = np.zeros((n, k), dtype=bool)
    aware_mask = (aware_series == 1).to_numpy()
    for j, cep in enumerate(cep_idx):
        s = raw[f"RS8_{cep}_1NET"]
        elig[:, j] = aware_mask # Only people who are aware can 'flip' to having an association
        X[:, j] = (s == 1).astype(int).to_numpy()
    return X, elig

def simulate_unique_reach(X, elig, w, salience_current_w, uplifts_pts, n_sims, cap, seed=7):
    rng = np.random.default_rng(seed)
    uplift = uplifts_pts / 100.0
    s_target = np.minimum(salience_current_w + uplift, cap)
    
    # Probability to flip is based on the remaining 'space' among aware people
    denom = (1.0 - salience_current_w)
    need = (s_target - salience_current_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        flip_prob = np.where(denom > 1e-9, need / denom, 0.0)
    flip_prob = np.clip(flip_prob, 0.0, 1.0)
    
    n, k = X.shape
    reach_dist = np.zeros(n_sims, dtype=float)
    for t in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        flips = elig & (X_sim == 0) & (U < flip_prob)
        X_sim[flips] = 1
        mpen_sim = (X_sim.max(axis=1) > 0).astype(int)
        reach_dist[t] = wmean(mpen_sim, w)
    return float(reach_dist.mean()), reach_dist, s_target

# ============================================================
# Main App
# ============================================================
def main():
    st.set_page_config(page_title="TFM CEP Simulator", layout="wide")
    st.title("TFM CEP Simulator — Dashboard-aligned (Weighted)")

    st.markdown("""
    **Brand Awareness Logic Applied:**
    - Baseline values are now shown as **% of Aware Respondents** (to match your tracker).
    - Market Penetration calculations still scale to the **Total Market**.
    """)

    st.sidebar.header("Data inputs")
    raw_all = load_csv_from_repo_or_upload("Raw data CSV", RAW_DATA_DEFAULT_NAMES)
    
    if AWARE_COL not in raw_all.columns:
        st.error(f"Awareness column '{AWARE_COL}' not found.")
        st.stop()

    st.sidebar.header("Universe controls")
    use_msa_only = st.sidebar.checkbox(f"Use MSAs only ({MSA_FILTER_COL}==1)", value=True)
    raw = apply_msa_filter(raw_all, use_msa_only)
    w = raw[WEIGHT_COL].to_numpy()

    cep_idx = discover_ceps(raw)
    labels = {i: CEP_NAME_MAP.get(i, f"CEP {i}") for i in cep_idx}
    
    # Calculate using Awareness denominator
    prevalence_w = np.array([weighted_category_prevalence(raw[f"RS1_{cep}NET"], w) for cep in cep_idx])
    salience_w = np.array([weighted_brand_salience(raw[f"RS8_{cep}_1NET"], raw[AWARE_COL], w) for cep in cep_idx])

    X, elig = build_X_and_eligibility(raw, raw[AWARE_COL], cep_idx)
    unique_reach_current_w = wmean((X.max(axis=1) > 0).astype(int), w)

    st.sidebar.header("Simulation controls")
    hh_base = st.sidebar.number_input("Household base", value=int(HOUSEHOLD_BASE_TFM_STATES), step=100000)
    n_sims = st.sidebar.slider("Monte Carlo runs", 200, 3000, DEFAULT_N_SIMS, step=100)
    
    st.sidebar.subheader("TFM salience uplifts (pts)")
    uplifts = np.zeros(len(cep_idx), dtype=float)
    order = np.argsort(-prevalence_w)
    ordered_ceps = [cep_idx[i] for i in order]

    for cep in ordered_ceps:
        j = cep_idx.index(cep)
        # These labels will now show 34% etc. instead of 7.5%
        uplifts[j] = st.sidebar.slider(f"{labels[cep]} (Current: {salience_w[j]*100:.1f}%)", 0, 50, 0)

    unique_reach_scenario_w, reach_dist_w, salience_target_w = simulate_unique_reach(
        X, elig, w, salience_w, uplifts, n_sims, DEFAULT_CAP
    )

    unique_hh_current = unique_reach_current_w * hh_base
    unique_hh_scenario = unique_reach_scenario_w * hh_base
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Market Penetration", f"{unique_reach_current_w:.1%}")
    c2.metric("Scenario Market Penetration", f"{unique_reach_scenario_w:.1%}", f"{(unique_reach_scenario_w - unique_reach_current_w):.1%}")
    c3.metric("New Households Gained", f"{unique_hh_scenario - unique_hh_current:,.0f}")

    # Results Table
    df = pd.DataFrame({
        "CEP": [labels[i] for i in cep_idx],
        "CEP Type": [infer_cep_type(labels[i]) for i in cep_idx],
        "Category salience (wtd %)": prevalence_w * 100,
        "TFM salience (Current %)": salience_w * 100,
        "TFM salience (Scenario %)": salience_target_w * 100,
        "Uplift (pts)": uplifts,
        "Accessible TAM (HHs)": prevalence_w * hh_base,
    }).sort_values("Category salience (wtd %)", ascending=False)

    st.subheader("CEP Metrics (Based on Brand Aware respondents)")
    st.dataframe(df, use_container_width=True)

    # Bubble Matrix with trails
    st.subheader("Bubble Matrix: Current vs Scenario")
    
    current_layer = alt.Chart(df).mark_circle(opacity=0.7, stroke="black", strokeWidth=1).encode(
        x=alt.X("Category salience (wtd %)", title="Category Prevalence (%)"),
        y=alt.Y("TFM salience (Current %)", title="TFM Brand Salience among Aware (%)"),
        size=alt.Size("Accessible TAM (HHs)", scale=alt.Scale(range=[120, 3000])),
        color=alt.Color("CEP Type"),
        tooltip=["CEP", "TFM salience (Current %)", "Accessible TAM (HHs)"]
    )

    scenario_layer = alt.Chart(df).mark_circle(opacity=0.3).encode(
        x=alt.X("Category salience (wtd %)"),
        y=alt.Y("TFM salience (Scenario %)"),
        size=alt.Size("Accessible TAM (HHs)", legend=None),
        color=alt.Color("CEP Type", legend=None)
    )

    lines = alt.Chart(df).mark_rule(color="gray", strokeDash=[4,4], opacity=0.5).encode(
        x=alt.X("Category salience (wtd %)"),
        y=alt.Y("TFM salience (Current %)"),
        y2=alt.Y2("TFM salience (Scenario %)")
    )

    st.altair_chart((lines + current_layer + scenario_layer).properties(height=600), use_container_width=True)

if __name__ == "__main__":
    main()
