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

# Geography filter: 1 = in TFM-present MSA universe
MSA_FILTER_COL = "xdemAud1"
MSA_FILTER_VALUE = 1

# Weight column
WEIGHT_COL = "wts"

# Monte Carlo defaults
DEFAULT_N_SIMS = 800
DEFAULT_CAP = 0.90

APP_DIR = Path(__file__).parent

# ============================================================
# CEP NAME MAP (match your dashboard)
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


def to_binary_selected(series: pd.Series) -> np.ndarray:
    """
    Convert 1=Selected, 2=Not Selected (and NaN -> Not Selected) to 0/1.
    """
    s = series.fillna(2)
    return (s.astype(float) == 1.0).astype(int).to_numpy()


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
    if not cep_idx:
        raise ValueError("No CEPs found. Expected RS1_iNET and RS8_i_1NET.")
    return cep_idx


def rs8_asked_mask(series: pd.Series) -> pd.Series:
    """
    True where respondent was asked the RS8 item and gave a coded response (1 or 2).
    """
    return series.notna() & series.isin([1, 2])


def salience_raw_dashboard(rs8_series: pd.Series) -> float:
    """
    RAW dashboard-aligned salience = % selecting TFM among respondents who were asked (RS8 present).
    """
    mask = rs8_asked_mask(rs8_series)
    if mask.sum() == 0:
        return 0.0
    return float((rs8_series[mask] == 1).mean())


def salience_wtd_dashboard(rs8_series: pd.Series, w: pd.Series) -> float:
    """
    Weighted version of the same: among asked respondents only.
    """
    mask = rs8_asked_mask(rs8_series)
    if mask.sum() == 0:
        return 0.0
    x = (rs8_series[mask] == 1).astype(int).to_numpy()
    ww = w[mask].to_numpy()
    return wmean(x, ww)


def simulate_unique_reach(
    X: np.ndarray,
    w: np.ndarray,
    salience_current_w: np.ndarray,
    uplifts_pts: np.ndarray,
    n_sims: int,
    cap: float,
    seed: int = 7,
):
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
    non = (X == 0)

    for t in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        flips = (U < flip_prob) & non
        X_sim[flips] = 1

        # Dedup: any CEP selected for TFM
        mpen_sim = (X_sim.max(axis=1) > 0).astype(int)
        reach_dist[t] = wmean(mpen_sim, w)

    return float(reach_dist.mean()), reach_dist, s_target


# ============================================================
# App
# ============================================================
def main():
    st.set_page_config(page_title="TFM CEP Simulator (Dashboard-aligned RAW)", layout="wide")
    st.title("TFM CEP Simulator — Dashboard-aligned RAW UI, Weighted + Deduped Engine")

    st.markdown(
        """
**Why your raw %s now match the dashboard:**  
The brand-association matrix is effectively measured on a **TFM-eligible base** (respondents who were asked the TFM column).  
So salience is calculated as **% selecting TFM among those asked** (RS8 coded 1/2), not diluted by blanks.
"""
    )

    st.sidebar.header("Data inputs")
    raw_all = load_csv_from_repo_or_upload("Raw data CSV", RAW_DATA_DEFAULT_NAMES, required=True)

    if WEIGHT_COL not in raw_all.columns:
        st.error(f"Missing required column: {WEIGHT_COL}")
        st.stop()

    st.sidebar.header("Universe controls")
    use_msa_only = st.sidebar.checkbox(
        f"Use TFM-present MSAs only ({MSA_FILTER_COL}=={MSA_FILTER_VALUE})",
        value=True
    )

    raw_geo = apply_msa_filter(raw_all, use_msa_only)

    # Discover CEPs
    cep_idx = discover_ceps(raw_geo)
    labels = {i: CEP_NAME_MAP.get(i, f"CEP {i}") for i in cep_idx}
    k = len(cep_idx)

    # -------------------------------------------------------------------
    # IMPORTANT: define the TFM-eligible base (asked base)
    # We use RS8 for CEP1 as a proxy: if you were asked once, you’re in the matrix base.
    # This works because in your file, RS8 and RS8mpen_1 are populated on the same base.
    # -------------------------------------------------------------------
    base_proxy_col = f"RS8_{cep_idx[0]}_1NET"
    asked_base_mask = rs8_asked_mask(raw_geo[base_proxy_col])

    raw = raw_geo[asked_base_mask].copy()

    # Weights
    w_all = raw_all[WEIGHT_COL].to_numpy()
    w_geo = raw_geo[WEIGHT_COL].to_numpy()
    w = raw[WEIGHT_COL].to_numpy()

    # Suggest HH base:
    # 1) scale 22-state HH base -> selected geography (MSA) using weighted share
    share_geo = float(w_geo.sum() / w_all.sum()) if w_all.sum() > 0 else 1.0
    hh_geo = int(round(HOUSEHOLD_BASE_TFM_STATES * share_geo))

    # 2) within selected geography, scale to asked/eligible base
    share_asked_within_geo = float(raw[WEIGHT_COL].sum() / raw_geo[WEIGHT_COL].sum()) if raw_geo[WEIGHT_COL].sum() > 0 else 1.0
    hh_asked = int(round(hh_geo * share_asked_within_geo))

    if use_msa_only:
        st.success(f"Universe: TFM-present MSAs ({MSA_FILTER_COL}=={MSA_FILTER_VALUE})")
    else:
        st.info("Universe: Full sample")

    st.caption(
        f"Geography share (weighted): **{share_geo:.1%}** → est. HH in geography: **{hh_geo:,}**. "
        f"TFM-eligible share within geography: **{share_asked_within_geo:.1%}** → est. eligible HH base: **{hh_asked:,}**."
    )

    st.sidebar.header("Simulation controls")
    hh_base = st.sidebar.number_input(
        "Household base (eligible base, for scaling to HHs)",
        value=int(hh_asked),
        step=100000
    )
    n_sims = st.sidebar.slider("Monte Carlo runs", 200, 3000, DEFAULT_N_SIMS, step=100)
    cap = st.sidebar.slider("Salience cap", 0.50, 0.95, DEFAULT_CAP, step=0.01)

    # RAW category prevalence (still meaningful on eligible base; if you prefer, we can compute prevalence on geo base instead)
    prevalence_raw = np.array([(raw_geo[f"RS1_{cep}NET"] == 1).mean() for cep in cep_idx])

    # RAW dashboard salience (asked base)
    salience_raw = np.array([salience_raw_dashboard(raw[f"RS8_{cep}_1NET"]) for cep in cep_idx])

    # Weighted dashboard salience (engine)
    salience_w = np.array([salience_wtd_dashboard(raw[f"RS8_{cep}_1NET"], raw[WEIGHT_COL]) for cep in cep_idx])

    # Weighted prevalence for TAM sizing (use geo base for prevalence sizing)
    prevalence_w_geo = np.array([
        wmean((raw_geo[f"RS1_{cep}NET"] == 1).astype(int).to_numpy(), raw_geo[WEIGHT_COL].to_numpy())
        for cep in cep_idx
    ])

    # Build X (no missing now; asked base only)
    X = np.column_stack([
        (raw[f"RS8_{cep}_1NET"].astype(float) == 1.0).astype(int).to_numpy()
        for cep in cep_idx
    ])

    # Baseline deduped reach (weighted) on eligible base
    if "RS8mpen_1" in raw.columns:
        mpen = (raw["RS8mpen_1"] == 1).astype(int).to_numpy()
        unique_reach_current_w = wmean(mpen, w)
    else:
        unique_reach_current_w = wmean((X.max(axis=1) > 0).astype(int), w)

    # Sidebar sliders (RAW only)
    st.sidebar.subheader("TFM salience uplifts (pts)")
    uplifts = np.zeros(k, dtype=float)

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

    # Simulate (weighted engine)
    unique_reach_scenario_w, reach_dist_w, salience_target_w = simulate_unique_reach(
        X=X,
        w=w,
        salience_current_w=salience_w,
        uplifts_pts=uplifts,
        n_sims=n_sims,
        cap=cap,
        seed=7
    )

    # KPIs in HH
    unique_hh_current = unique_reach_current_w * hh_base
    unique_hh_scenario = unique_reach_scenario_w * hh_base
    delta_unique_hh = unique_hh_scenario - unique_hh_current

    lo = float(np.quantile(reach_dist_w, 0.10)) * hh_base
    hi = float(np.quantile(reach_dist_w, 0.90)) * hh_base

    c1, c2, c3 = st.columns(3)
    c1.metric("Unique HH Reach (current, deduped)", f"{unique_hh_current:,.0f}")
    c2.metric("Unique HH Reach (scenario, deduped)", f"{unique_hh_scenario:,.0f}", f"{delta_unique_hh:,.0f}")
    c3.metric("Scenario range (P10–P90)", f"{lo:,.0f} – {hi:,.0f}")

    # Table (raw-facing salience, geo-based prevalence for TAM)
    df = pd.DataFrame({
        "CEP": [labels[i] for i in cep_idx],
        "CEP Type": [infer_cep_type(labels[i]) for i in cep_idx],
        "Category prevalence (raw %, geo base)": prevalence_raw * 100,
        "TFM salience (raw %, asked base)": salience_raw * 100,
        "Uplift (pts)": uplifts,
        "Accessible TAM (HHs, geo)": prevalence_w_geo * hh_geo,
    }).sort_values("Category prevalence (raw %, geo base)", ascending=False)

    st.subheader("Dashboard-aligned view")
    st.caption("Salience is computed on the **asked/eligible base** (matches your dashboard). TAM is sized on the **geo base**.")
    st.dataframe(df, use_container_width=True)

    # Bubble chart (raw axes, HH sizing)
    st.subheader("Bubble matrix")
    chart_df = df.copy()

    bubble = (
        alt.Chart(chart_df)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("Category prevalence (raw %, geo base)", title="Category prevalence (raw %, geo base)"),
            y=alt.Y("TFM salience (raw %, asked base)", title="TFM salience (raw %, asked base)"),
            size=alt.Size("Accessible TAM (HHs, geo)", title="Accessible TAM (HHs)", scale=alt.Scale(range=[120, 3200])),
            color=alt.Color("CEP Type", title="CEP Type"),
            tooltip=[
                "CEP",
                "CEP Type",
                alt.Tooltip("Category prevalence (raw %, geo base)", format=".1f"),
                alt.Tooltip("TFM salience (raw %, asked base)", format=".1f"),
                alt.Tooltip("Accessible TAM (HHs, geo)", format=",.0f"),
            ],
        )
        .properties(height=520)
    )

    x_med = float(chart_df["Category prevalence (raw %, geo base)"].median())
    y_med = float(chart_df["TFM salience (raw %, asked base)"].median())
    vline = alt.Chart(pd.DataFrame({"x": [x_med]})).mark_rule(strokeDash=[4, 4]).encode(x="x")
    hline = alt.Chart(pd.DataFrame({"y": [y_med]})).mark_rule(strokeDash=[4, 4]).encode(y="y")

    st.altair_chart(bubble + vline + hline, use_container_width=True)
    st.caption(f"Dotted lines are medians: x={x_med:.1f}%, y={y_med:.1f}%.")


if __name__ == "__main__":
    main()

