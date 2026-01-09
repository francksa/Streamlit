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

# Household base for 22-state footprint (2023)
HOUSEHOLD_BASE_TFM_STATES = 70_132_819

# Universe filter: 1 = in TFM-present MSA universe
MSA_FILTER_COL = "xdemAud1"
MSA_FILTER_VALUE = 1

# Weight column
WEIGHT_COL = "wts"

# Monte Carlo defaults
DEFAULT_N_SIMS = 800
DEFAULT_CAP = 0.90

APP_DIR = Path(__file__).parent

# ============================================================
# CEP NAME MAP (matches your dashboard wording)
# ============================================================
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
    """
    Find CEP indices i where BOTH exist:
      - RS1_{i}NET
      - RS8_{i}_1NET  (TFM = brand 1)
    """
    cep_idx = []
    for col in raw.columns:
        m = re.match(r"RS1_(\d+)NET$", str(col))
        if m:
            i = int(m.group(1))
            if f"RS8_{i}_1NET" in raw.columns:
                cep_idx.append(i)
    cep_idx = sorted(set(cep_idx))
    if not cep_idx:
        raise ValueError("No CEPs found. Expected RS1_iNET and RS8_i_1NET.")
    return cep_idx


def asked_mask(series: pd.Series) -> np.ndarray:
    """
    Respondents who were asked / eligible for this RS8 item.
    Assumes 1=Selected, 2=Not selected. Missing = not asked / not eligible.
    """
    return (series.notna() & series.isin([1, 2])).to_numpy()


def weighted_category_prevalence(rs1_series: pd.Series, w: np.ndarray) -> float:
    """
    Dashboard-aligned: weighted % selecting CEP in this universe.
    """
    x = (rs1_series == 1).astype(int).to_numpy()
    return wmean(x, w)


def weighted_brand_salience(rs8_series: pd.Series, w: np.ndarray) -> float:
    """
    Dashboard-aligned: weighted % selecting TFM among eligible/asked respondents for this CEP.
    """
    m = asked_mask(rs8_series)
    if m.sum() == 0:
        return 0.0
    x = (rs8_series[m] == 1).astype(int).to_numpy()
    ww = w[m]
    return wmean(x, ww)


def build_X_and_eligibility(raw: pd.DataFrame, cep_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds:
      X: n x k binary matrix of current TFM associations (1 if selected, else 0).
      elig: n x k boolean matrix where True means respondent was asked/eligible for that CEP’s brand selection.
    Missing values become 0 in X but False in elig (and won't be flipped by uplift).
    """
    n = len(raw)
    k = len(cep_idx)
    X = np.zeros((n, k), dtype=int)
    elig = np.zeros((n, k), dtype=bool)

    for j, cep in enumerate(cep_idx):
        s = raw[f"RS8_{cep}_1NET"]
        m = asked_mask(s)
        elig[:, j] = m
        # among eligible: 1->1, 2->0
        X[m, j] = (s[m] == 1).astype(int).to_numpy()

    return X, elig


def simulate_unique_reach(
    X: np.ndarray,
    elig: np.ndarray,
    w: np.ndarray,
    salience_current_w: np.ndarray,
    uplifts_pts: np.ndarray,
    n_sims: int,
    cap: float,
    seed: int = 7,
):
    """
    Weighted + deduped Monte Carlo.
    Uplift only applies within the eligible/asked base for each CEP (elig[:, j] == True).
    """
    rng = np.random.default_rng(seed)

    uplift = uplifts_pts / 100.0
    s = salience_current_w
    s_target = np.minimum(s + uplift, cap)

    denom = (1.0 - s)
    need = (s_target - s)
    with np.errstate(divide="ignore", invalid="ignore"):
        flip_prob = np.where(denom > 1e-9, need / denom, 0.0)
    flip_prob = np.clip(flip_prob, 0.0, 1.0)

    n, k = X.shape
    reach_dist = np.zeros(n_sims, dtype=float)

    for t in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))

        # Only flip: eligible AND currently 0 AND random < flip_prob
        flips = elig & (X_sim == 0) & (U < flip_prob)
        X_sim[flips] = 1

        # Dedup: any CEP selected for TFM
        mpen_sim = (X_sim.max(axis=1) > 0).astype(int)
        reach_dist[t] = wmean(mpen_sim, w)

    return float(reach_dist.mean()), reach_dist, s_target


# ============================================================
# Streamlit App
# ============================================================
def main():
    st.set_page_config(page_title="TFM CEP Simulator (Dashboard-aligned Weighted)", layout="wide")
    st.title("TFM CEP Simulator — Dashboard-aligned (Weighted)")

    st.markdown(
        """
This app is aligned to your dashboard definitions:

- **Category CEP Salience** = **weighted %** selecting each CEP within the selected universe  
- **TFM CEP Salience** = **weighted % selecting TFM among eligible/asked respondents for that CEP**  
- **Simulation** = weighted + deduped Monte Carlo; uplift only acts within the eligible base (no “creating awareness”)
"""
    )

    # ----------------------------
    # Data input
    # ----------------------------
    st.sidebar.header("Data inputs")
    raw_all = load_csv_from_repo_or_upload("Raw data CSV", RAW_DATA_DEFAULT_NAMES, required=True)

    if WEIGHT_COL not in raw_all.columns:
        st.error(f"Missing required column: {WEIGHT_COL}")
        st.stop()

    # ----------------------------
    # Universe controls
    # ----------------------------
    st.sidebar.header("Universe controls")
    use_msa_only = st.sidebar.checkbox(
        f"Use TFM-present MSAs only ({MSA_FILTER_COL}=={MSA_FILTER_VALUE})",
        value=True
    )

    raw = apply_msa_filter(raw_all, use_msa_only)

    # suggested HH base: scale 22-state HH base by weighted share in selected universe
    w_all_sum = float(raw_all[WEIGHT_COL].sum())
    w_sel_sum = float(raw[WEIGHT_COL].sum())
    share_in_universe = (w_sel_sum / w_all_sum) if w_all_sum > 0 else 1.0
    hh_base_suggested = int(round(HOUSEHOLD_BASE_TFM_STATES * share_in_universe))

    if use_msa_only:
        st.success(f"Universe: TFM-present MSAs ({MSA_FILTER_COL}=={MSA_FILTER_VALUE})")
    else:
        st.info("Universe: Full sample")

    st.caption(
        f"Weighted share in selected universe: **{share_in_universe:.1%}**. "
        f"Suggested HH base: **{hh_base_suggested:,}** (scaled from {HOUSEHOLD_BASE_TFM_STATES:,} HH in TFM states)."
    )

    # weights
    w = raw[WEIGHT_COL].to_numpy()

    # ----------------------------
    # CEPs + labels
    # ----------------------------
    cep_idx = discover_ceps(raw)
    labels = {i: CEP_NAME_MAP.get(i, f"CEP {i}") for i in cep_idx}
    k = len(cep_idx)

    # ----------------------------
    # Dashboard-aligned metrics (WEIGHTED)
    # ----------------------------
    prevalence_w = np.array([weighted_category_prevalence(raw[f"RS1_{cep}NET"], w) for cep in cep_idx])
    salience_w = np.array([weighted_brand_salience(raw[f"RS8_{cep}_1NET"], w) for cep in cep_idx])

    # ----------------------------
    # Build X + eligibility for sim
    # ----------------------------
    X, elig = build_X_and_eligibility(raw, cep_idx)

    # Baseline deduped reach (weighted): prefer RS8mpen_1 if present
    if "RS8mpen_1" in raw.columns:
        mpen = (raw["RS8mpen_1"] == 1).astype(int).to_numpy()
        unique_reach_current_w = wmean(mpen, w)
    else:
        unique_reach_current_w = wmean((X.max(axis=1) > 0).astype(int), w)

    # ----------------------------
    # Sidebar simulation controls
    # ----------------------------
    st.sidebar.header("Simulation controls")
    hh_base = st.sidebar.number_input(
        "Household base (for scaling to HHs)",
        value=int(hh_base_suggested if use_msa_only else HOUSEHOLD_BASE_TFM_STATES),
        step=100000
    )
    n_sims = st.sidebar.slider("Monte Carlo runs", 200, 3000, DEFAULT_N_SIMS, step=100)
    cap = st.sidebar.slider("Salience cap", 0.50, 0.95, DEFAULT_CAP, step=0.01)

    st.sidebar.subheader("TFM salience uplifts (pts)")
    uplifts = np.zeros(k, dtype=float)

    # order sliders by weighted category prevalence (dashboard-like)
    order = np.argsort(-prevalence_w)
    ordered_ceps = [cep_idx[i] for i in order]

    for cep in ordered_ceps:
        j = cep_idx.index(cep)
        uplifts[j] = st.sidebar.slider(
            f"{labels[cep]} (current {salience_w[j]*100:.0f}% wtd)",
            min_value=-10,
            max_value=25,
            value=0,
            step=1,
        )

    # ----------------------------
    # Run simulation
    # ----------------------------
    unique_reach_scenario_w, reach_dist_w, salience_target_w = simulate_unique_reach(
        X=X,
        elig=elig,
        w=w,
        salience_current_w=salience_w,
        uplifts_pts=uplifts,
        n_sims=n_sims,
        cap=cap,
        seed=7
    )

    # KPIs (scaled)
    unique_hh_current = unique_reach_current_w * hh_base
    unique_hh_scenario = unique_reach_scenario_w * hh_base
    delta_unique_hh = unique_hh_scenario - unique_hh_current

    lo = float(np.quantile(reach_dist_w, 0.10)) * hh_base
    hi = float(np.quantile(reach_dist_w, 0.90)) * hh_base

    c1, c2, c3 = st.columns(3)
    c1.metric("Unique HH Reach (current, deduped)", f"{unique_hh_current:,.0f}")
    c2.metric("Unique HH Reach (scenario, deduped)", f"{unique_hh_scenario:,.0f}", f"{delta_unique_hh:,.0f}")
    c3.metric("Scenario range (P10–P90)", f"{lo:,.0f} – {hi:,.0f}")

    # ----------------------------
    # Table (dashboard-aligned)
    # ----------------------------
    df = pd.DataFrame({
        "CEP": [labels[i] for i in cep_idx],
        "CEP Type": [infer_cep_type(labels[i]) for i in cep_idx],
        "Category salience (wtd %)": prevalence_w * 100,
        "TFM salience (wtd %, eligible base)": salience_w * 100,
        "Uplift (pts)": uplifts,
        "Accessible TAM (HHs)": prevalence_w * hh_base,
    }).sort_values("Category salience (wtd %)", ascending=False)

    st.subheader("Dashboard-aligned CEP metrics (weighted)")
    st.dataframe(df, use_container_width=True)

    # ----------------------------
    # Bubble chart (weighted axes)
    # ----------------------------
    st.subheader("Bubble matrix (weighted, dashboard-aligned)")
    st.caption("X = Category CEP salience (weighted). Y = TFM CEP salience (weighted, eligible base). Bubble size = Accessible TAM (HHs).")

    chart_df = df.copy()

    bubble = (
        alt.Chart(chart_df)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("Category salience (wtd %)", title="Category CEP salience (weighted %)"),
            y=alt.Y("TFM salience (wtd %, eligible base)", title="TFM CEP salience (weighted %, eligible base)"),
            size=alt.Size("Accessible TAM (HHs)", title="Accessible TAM (HHs)", scale=alt.Scale(range=[120, 3200])),
            color=alt.Color("CEP Type", title="CEP Type"),
            tooltip=[
                "CEP",
                "CEP Type",
                alt.Tooltip("Category salience (wtd %)", format=".1f"),
                alt.Tooltip("TFM salience (wtd %, eligible base)", format=".1f"),
                alt.Tooltip("Accessible TAM (HHs)", format=",.0f"),
            ],
        )
        .properties(height=520)
    )

    x_med = float(chart_df["Category salience (wtd %)"].median())
    y_med = float(chart_df["TFM salience (wtd %, eligible base)"].median())
    vline = alt.Chart(pd.DataFrame({"x": [x_med]})).mark_rule(strokeDash=[4, 4]).encode(x="x")
    hline = alt.Chart(pd.DataFrame({"y": [y_med]})).mark_rule(strokeDash=[4, 4]).encode(y="y")

    st.altair_chart(bubble + vline + hline, use_container_width=True)
    st.caption(f"Dotted lines are medians (weighted): x={x_med:.1f}%, y={y_med:.1f}%.")


if __name__ == "__main__":
    main()


