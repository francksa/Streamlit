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

# MSA filter: 1 = in TFM-present MSA universe
MSA_FILTER_COL = "xdemAud1"
MSA_FILTER_VALUE = 1

# Weight column
WEIGHT_COL = "wts"

# Monte Carlo defaults
DEFAULT_N_SIMS = 800
DEFAULT_CAP = 0.90

APP_DIR = Path(__file__).parent

# ============================================================
# CEP NAME MAP (from your dashboard / workbook)
# Assumption (as you stated): RS1_1NET corresponds to "Doing my regular weekly grocery shopping", etc.
# Edit wording here if you want it to match your dashboard text exactly.
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

# Optional CEP type categorization (for bubble colors)
def infer_cep_type(label: str) -> str:
    s = label.lower()
    routine_kw = ["weekly", "regular", "save money", "low on supplies", "planning", "errands"]
    health_kw = ["healthy", "health-conscious", "eat better", "better-quality", "environment"]
    convenience_kw = ["short on time", "convenient", "ready-to-eat", "pickup", "online", "crowded", "long lines", "avoid"]
    inspiration_kw = ["inspired", "try something new", "new or seasonal", "seasonal", "hosting", "special meal", "ideas", "help"]
    specialty_kw = ["specialty", "international", "large family", "group"]

    if any(k in s for k in routine_kw):
        return "Routine"
    if any(k in s for k in health_kw):
        return "Health"
    if any(k in s for k in convenience_kw):
        return "Convenience"
    if any(k in s for k in inspiration_kw):
        return "Inspiration"
    if any(k in s for k in specialty_kw):
        return "Specialty"
    return "Other"


# ============================================================
# Utilities
# ============================================================
def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.sum(w))


def to_binary_selected(series: pd.Series) -> np.ndarray:
    """
    Convert 1=Selected, 2=Not Selected (and NaN -> Not Selected) to 0/1.
    IMPORTANT: This is correct for RS1 (prevalence) and RS8 grids.
    """
    s = series.fillna(2)
    return (s.astype(float) == 1.0).astype(int).to_numpy()


def load_csv_from_repo_or_upload(label: str, default_names: list[str], required: bool = True):
    """
    Tries to load from repo; otherwise falls back to uploader.
    """
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
      - RS1_{i}NET  (category prevalence item)
      - RS8_{i}_1NET (TFM association for CEP i; brand 1 = TFM)
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
        raise ValueError("No CEPs found. Expected columns like RS1_iNET and RS8_i_1NET.")
    return cep_idx


def conditional_share_raw(x_col: np.ndarray, cep_col: np.ndarray) -> float:
    """
    Raw salience = P(TFM | CEP) among CEP-doers only.
    """
    mask = (cep_col == 1)
    if mask.sum() == 0:
        return 0.0
    return float(x_col[mask].mean())


def conditional_share_wtd(x_col: np.ndarray, cep_col: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted salience = P(TFM | CEP) among CEP-doers only (weights applied within mask).
    """
    mask = (cep_col == 1)
    if mask.sum() == 0:
        return 0.0
    return wmean(x_col[mask], w[mask])


def simulate_unique_reach(
    X: np.ndarray,
    w: np.ndarray,
    salience_current_w: np.ndarray,
    uplifts_pts: np.ndarray,
    n_sims: int,
    cap: float,
    seed: int = 7,
):
    """
    Respondent-level Monte Carlo with weights + true deduplication.

    NOTE:
    - salience_current_w is WEIGHTED and conditional on CEP-doers.
    - We treat uplift as moving the conditional salience for that CEP among CEP-doers.
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

        # Flip only among current non-selectors within EACH CEP column
        U = rng.random((n, k))
        flips = (U < flip_prob) & (X_sim == 0)
        X_sim[flips] = 1

        # Dedup: any CEP selected for TFM
        mpen_sim = (X_sim.max(axis=1) > 0).astype(int)
        reach_dist[t] = wmean(mpen_sim, w)

    return float(reach_dist.mean()), reach_dist, s_target


# ============================================================
# Streamlit App
# ============================================================
def main():
    st.set_page_config(page_title="TFM CEP Simulator (Raw UI, Weighted Engine)", layout="wide")
    st.title("TFM CEP Simulator — Raw Dashboard UI, Weighted + Deduped Engine")

    st.markdown(
        """
**What you see (UI):** Raw (unweighted) *conditional* associations that match your dashboard  
- Category prevalence: \(P(CEP)\)  
- TFM salience: \(P(TFM \\mid CEP)\) among CEP-doers

**What the simulator uses (engine):** Weighted + deduped Monte Carlo  
- Scenario HH impact is overlap-safe and population-scaled.
"""
    )

    # ----------------------------
    # Sidebar: data inputs
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

    try:
        raw = apply_msa_filter(raw_all, use_msa_only)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Suggested HH base for selected universe (scale from 22-state HH base)
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

    # ----------------------------
    # CEP discovery + labeling
    # ----------------------------
    cep_idx = discover_ceps(raw)
    labels = {i: CEP_NAME_MAP.get(i, f"CEP {i}") for i in cep_idx}
    k = len(cep_idx)
    st.caption(f"Detected **{k} CEPs** with both RS1 prevalence and RS8 TFM association data in the selected universe.")

    # Build matrices
    prev_cols = [f"RS1_{i}NET" for i in cep_idx]
    tfm_cols = [f"RS8_{i}_1NET" for i in cep_idx]

    prev_mat = np.column_stack([to_binary_selected(raw[c]) for c in prev_cols])  # 0/1
    X = np.column_stack([to_binary_selected(raw[c]) for c in tfm_cols])          # 0/1

    w = raw[WEIGHT_COL].to_numpy()

    # ----------------------------
    # RAW (dashboard) metrics
    # ----------------------------
    prevalence_raw = np.array([prev_mat[:, j].mean() for j in range(k)])  # P(CEP)
    # conditional raw salience: P(TFM | CEP)
    salience_raw = np.array([conditional_share_raw(X[:, j], prev_mat[:, j]) for j in range(k)])

    # ----------------------------
    # WEIGHTED (engine) metrics
    # ----------------------------
    prevalence_w = np.array([wmean(prev_mat[:, j], w) for j in range(k)])  # weighted P(CEP)
    # conditional weighted salience: P(TFM | CEP) among CEP-doers
    salience_w = np.array([conditional_share_wtd(X[:, j], prev_mat[:, j], w) for j in range(k)])

    # Baseline deduped reach (weighted) – prefer RS8mpen_1 if present
    if "RS8mpen_1" in raw.columns:
        mpen = to_binary_selected(raw["RS8mpen_1"])
        unique_reach_current_w = wmean(mpen, w)
        # validation from X union
        mpen_from_x = (X.max(axis=1) > 0).astype(int)
        unique_reach_from_x_w = wmean(mpen_from_x, w)
    else:
        mpen_from_x = (X.max(axis=1) > 0).astype(int)
        unique_reach_current_w = wmean(mpen_from_x, w)
        unique_reach_from_x_w = unique_reach_current_w

    # ----------------------------
    # Sidebar: simulation controls
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
    st.sidebar.caption("Current salience shown below is RAW and conditional on CEP (matches dashboard).")

    uplifts = np.zeros(k, dtype=float)

    # order sliders by RAW category prevalence (dashboard-friendly)
    order = np.argsort(-prevalence_raw)
    ordered_ceps = [cep_idx[i] for i in order]

    for cep in ordered_ceps:
        j = cep_idx.index(cep)
        uplifts[j] = st.sidebar.slider(
            f"{labels[cep]} (current {salience_raw[j]*100:.0f}% raw)",
            min_value=-10,
            max_value=25,
            value=0,
            step=1,
        )

    # ----------------------------
    # Simulation (weighted engine)
    # ----------------------------
    unique_reach_scenario_w, reach_dist_w, salience_target_w = simulate_unique_reach(
        X=X,
        w=w,
        salience_current_w=salience_w,
        uplifts_pts=uplifts,
        n_sims=n_sims,
        cap=cap,
        seed=7
    )

    # KPIs (scaled to HH base)
    unique_hh_current = unique_reach_current_w * hh_base
    unique_hh_scenario = unique_reach_scenario_w * hh_base
    delta_unique_hh = unique_hh_scenario - unique_hh_current

    lo = float(np.quantile(reach_dist_w, 0.10)) * hh_base
    hi = float(np.quantile(reach_dist_w, 0.90)) * hh_base

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique HH Reach (current, deduped)", f"{unique_hh_current:,.0f}")
    c2.metric("Unique HH Reach (scenario, deduped)", f"{unique_hh_scenario:,.0f}", f"{delta_unique_hh:,.0f}")
    c3.metric("Scenario range (P10–P90)", f"{lo:,.0f} – {hi:,.0f}")
    diff_pp = (unique_reach_from_x_w - unique_reach_current_w) * 100
    c4.metric("Dedup validation (X vs RS8mpen_1)", f"{diff_pp:+.2f} pp")

    st.caption(
        "UI %s are RAW (unweighted) and conditional on CEP (dashboard-aligned). "
        "Scenario math is WEIGHTED + DEDUPED."
    )

    # ----------------------------
    # Table (raw-facing, with weighted sizing behind it)
    # ----------------------------
    df = pd.DataFrame({
        "CEP Index": cep_idx,
        "CEP": [labels[i] for i in cep_idx],
        "CEP Type": [infer_cep_type(labels[i]) for i in cep_idx],

        # Dashboard-aligned
        "Category prevalence (raw %)": prevalence_raw * 100,
        "TFM salience (raw %, P(TFM|CEP))": salience_raw * 100,

        # Engine transparency (kept, but you can remove these columns if you want)
        "Category prevalence (wtd %)": prevalence_w * 100,
        "TFM salience (wtd current %, P(TFM|CEP))": salience_w * 100,
        "TFM salience (wtd scenario %, P(TFM|CEP))": salience_target_w * 100,

        "Uplift (pts)": uplifts,

        # Household sizing (weighted prevalence × HH base)
        "Accessible TAM (HHs)": prevalence_w * hh_base,

        # Diagnostic (not deduped)
        "Brand TAM current (HHs) [diagnostic]": (prevalence_w * salience_w) * hh_base,
        "Brand TAM scenario (HHs) [diagnostic]": (prevalence_w * salience_target_w) * hh_base,
        "Δ Brand TAM (HHs) [diagnostic]": ((prevalence_w * salience_target_w) - (prevalence_w * salience_w)) * hh_base,
    }).sort_values("Category prevalence (raw %)", ascending=False)

    st.subheader("CEP diagnostics")
    st.caption(
        "Brand TAM columns are not deduplicated (CEPs overlap by design). "
        "Top KPI is deduplicated Unique HH Reach."
    )
    st.dataframe(df, use_container_width=True)

    # ----------------------------
    # Bubble matrix (RAW axes; WEIGHTED bubble sizes)
    # ----------------------------
    st.subheader("CEP Opportunity Bubble Matrix (Scenario)")
    st.caption(
        "Axes are RAW % (dashboard-aligned). Bubble size = Accessible TAM (HHs) using weighted prevalence × HH base."
    )

    chart_df = df.copy()

    bubble = (
        alt.Chart(chart_df)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("Category prevalence (raw %)", title="Category prevalence (raw %)"),
            y=alt.Y("TFM salience (raw %, P(TFM|CEP))", title="TFM salience (raw %, P(TFM|CEP))"),
            size=alt.Size("Accessible TAM (HHs)", title="Accessible TAM (HHs)", scale=alt.Scale(range=[120, 3200])),
            color=alt.Color("CEP Type", title="CEP Type"),
            tooltip=[
                "CEP",
                "CEP Type",
                alt.Tooltip("Category prevalence (raw %)", format=".1f"),
                alt.Tooltip("TFM salience (raw %, P(TFM|CEP))", format=".1f"),
                alt.Tooltip("Accessible TAM (HHs)", format=",.0f"),
                alt.Tooltip("Δ Brand TAM (HHs) [diagnostic]", format=",.0f"),
            ],
        )
        .properties(height=520)
    )

    x_med = float(chart_df["Category prevalence (raw %)"].median())
    y_med = float(chart_df["TFM salience (raw %, P(TFM|CEP))"].median())
    vline = alt.Chart(pd.DataFrame({"x": [x_med]})).mark_rule(strokeDash=[4, 4]).encode(x="x")
    hline = alt.Chart(pd.DataFrame({"y": [y_med]})).mark_rule(strokeDash=[4, 4]).encode(y="y")

    st.altair_chart(bubble + vline + hline, use_container_width=True)
    st.caption(f"Dotted lines are medians (raw): x={x_med:.1f}%, y={y_med:.1f}%.")


if __name__ == "__main__":
    main()
