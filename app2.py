import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# CONFIG
# ============================================================
# If files are in repo, set these names; if not, uploader will kick in automatically.
RAW_DATA_DEFAULT_NAMES = [
    "2510110_research_science_raw_data (1).csv",
    "2510110_research_science_raw_data.csv",
]

LEVELS_CODEBOOK_DEFAULT_NAMES = [
    "2510110_research_science_levels_codebook.csv",
]

QUESTION_CODEBOOK_DEFAULT_NAMES = [
    "2510110_research_science_question_codebook - 2510110_research_science_question_codebook.csv",
]

# Household base for 22-state footprint (2023) you provided
HOUSEHOLD_BASE_TFM_STATES = 70_132_819

# Total US households (2023) for reference only (not used unless you choose)
HOUSEHOLD_BASE_US = 127_482_785

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
    """
    Tries to load a CSV from the app directory using one of default_names.
    If not found, asks user to upload via sidebar.
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
        else:
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
    Find CEP indices i where BOTH:
      - RS1_{i}NET exists (category prevalence)
      - RS8_{i}_1NET exists (TFM association; brand 1 = TFM)
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


def get_cep_labels(levels_cb: pd.DataFrame, question_cb: pd.DataFrame, cep_idx: list[int]) -> dict[int, str]:
    """
    Robust CEP label extraction.
    We will display these labels instead of CEP1/CEP2.

    Priority:
      1) Question codebook: RS1_{i}NET row, best text-like column
      2) Levels codebook: RS1_{i}NET code==1, 'value' (if not just "Selected")
      3) Fallback: "RS1_{i}NET"
    """
    labels = {i: f"RS1_{i}NET" for i in cep_idx}

    # ---- Question codebook first (best shot at actual wording) ----
    if question_cb is not None:
        qc = question_cb.copy()
        qc.columns = [c.strip().lower() for c in qc.columns]

        id_candidates = [c for c in qc.columns if c in ("qid", "question", "variable", "var", "name")]
        id_col = id_candidates[0] if id_candidates else None

        text_priority = ["question_text", "questiontext", "text", "label", "description", "prompt"]
        text_col = next((c for c in text_priority if c in qc.columns), None)

        if id_col and text_col:
            for i in cep_idx:
                key = f"RS1_{i}NET".upper()
                hit = qc[qc[id_col].astype(str).str.upper() == key]
                if len(hit) > 0:
                    candidate = str(hit.iloc[0][text_col]).strip()
                    if candidate and candidate.lower() != "nan":
                        labels[i] = candidate

    # ---- Levels codebook fallback ----
    if levels_cb is not None:
        cb = levels_cb.copy()
        cb.columns = [c.strip().lower() for c in cb.columns]
        if {"question", "code", "value"}.issubset(set(cb.columns)):
            for i in cep_idx:
                # Only overwrite if still generic
                if labels[i].startswith("RS1_"):
                    q = f"RS1_{i}NET"
                    hit = cb[(cb["question"].astype(str) == q) & (cb["code"].astype(str) == "1")]
                    if len(hit) > 0:
                        candidate = str(hit.iloc[0]["value"]).strip()
                        if candidate and candidate.lower() not in ("selected", "yes", "1", "true") and candidate.lower() != "nan":
                            labels[i] = candidate

    return labels


def infer_cep_type(label: str) -> str:
    """
    Optional categorization used only for bubble colors.
    """
    s = label.lower()

    routine_kw = ["weekly", "regular", "save money", "low on supplies", "errands", "planning meals", "meal plan"]
    health_kw = ["healthy", "health-conscious", "eat better", "better-quality", "environment"]
    convenience_kw = ["short on time", "convenient", "ready-to-eat", "pickup", "online", "crowded", "long lines", "avoid"]
    inspiration_kw = ["inspired", "try something new", "new or seasonal", "seasonal", "hosting", "special meal", "ideas"]
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

    salience_current_w is WEIGHTED (modeling layer).
    Slider labels are RAW only (display layer).
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
    non = (X == 0)

    for t in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        flips = (U < flip_prob) & non
        X_sim[flips] = 1

        # Dedup: any CEP selected
        mpen_sim = (X_sim.max(axis=1) > 0).astype(int)
        reach_dist[t] = wmean(mpen_sim, w)

    return float(reach_dist.mean()), reach_dist, s_target


# ============================================================
# Streamlit App
# ============================================================
def main():
    st.set_page_config(page_title="TFM CEP Simulator (Deduped, Weighted)", layout="wide")
    st.title("TFM CEP Simulator — Deduped Unique HH Reach (Weighted)")
    st.markdown(
        """
**Display layer:** Raw (unweighted) associations — aligned to your dashboard.  
**Model layer:** Weighted + deduplicated Monte Carlo simulation — used for HH impact and scenario math.
"""
    )

    # ----------------------------
    # Sidebar: data inputs
    # ----------------------------
    st.sidebar.header("Data inputs")
    raw_all = load_csv_from_repo_or_upload("Raw data CSV", RAW_DATA_DEFAULT_NAMES, required=True)
    levels_cb = load_csv_from_repo_or_upload("Levels codebook CSV (optional)", LEVELS_CODEBOOK_DEFAULT_NAMES, required=False)
    question_cb = load_csv_from_repo_or_upload("Question codebook CSV (optional)", QUESTION_CODEBOOK_DEFAULT_NAMES, required=False)

    # Validate required columns
    if WEIGHT_COL not in raw_all.columns:
        st.error(f"Missing required column: {WEIGHT_COL}")
        st.stop()

    # ----------------------------
    # Universe controls (MSA-only vs full)
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

    # Compute weighted share for suggested HH base
    w_all_sum = float(raw_all[WEIGHT_COL].sum())
    w_sel_sum = float(raw[WEIGHT_COL].sum())
    share_in_universe = (w_sel_sum / w_all_sum) if w_all_sum > 0 else 1.0

    # We scale from 22-state HH base because xdemAud1 is a footprint-within-footprint flag
    hh_base_suggested = int(round(HOUSEHOLD_BASE_TFM_STATES * share_in_universe))

    if use_msa_only:
        st.success(f"Universe: TFM-present MSAs ({MSA_FILTER_COL}=={MSA_FILTER_VALUE})")
    else:
        st.info("Universe: Full sample")

    st.caption(
        f"Weighted share in selected universe: **{share_in_universe:.1%}**. "
        f"Suggested HH base: **{hh_base_suggested:,}** (scaled from {HOUSEHOLD_BASE_TFM_STATES:,} households in TFM states)."
    )

    # ----------------------------
    # Discover CEPs + labels
    # ----------------------------
    cep_idx = discover_ceps(raw)
    labels = get_cep_labels(levels_cb, question_cb, cep_idx)
    k = len(cep_idx)

    st.caption(f"Detected **{k} CEPs** with both prevalence (RS1) and TFM association (RS8) data in the selected universe.")

    # Build matrices
    prev_cols = [f"RS1_{i}NET" for i in cep_idx]
    tfm_cols = [f"RS8_{i}_1NET" for i in cep_idx]

    prev_mat = np.column_stack([to_binary_selected(raw[c]) for c in prev_cols])  # 0/1
    X = np.column_stack([to_binary_selected(raw[c]) for c in tfm_cols])          # 0/1

    # Weights
    w = raw[WEIGHT_COL].to_numpy()

    # ----------------------------
    # Baselines: RAW (display) + WEIGHTED (model)
    # ----------------------------
    prevalence_raw = np.array([prev_mat[:, j].mean() for j in range(k)])
    salience_raw = np.array([X[:, j].mean() for j in range(k)])

    prevalence_w = np.array([wmean(prev_mat[:, j], w) for j in range(k)])
    salience_w = np.array([wmean(X[:, j], w) for j in range(k)])

    # Deduped baseline reach (weighted)
    if "RS8mpen_1" in raw.columns:
        mpen = to_binary_selected(raw["RS8mpen_1"])
        unique_reach_current_w = wmean(mpen, w)
        # Also compute from X to validate
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
    st.sidebar.caption("Shown as RAW (unweighted) current salience to match the dashboard.")

    uplifts = np.zeros(k, dtype=float)

    # Sort sliders by RAW category prevalence (dashboard-friendly ordering)
    order = np.argsort(-prevalence_raw)
    ordered_ceps = [cep_idx[i] for i in order]

    for cep in ordered_ceps:
        j = cep_idx.index(cep)
        uplifts[j] = st.sidebar.slider(
            f"{labels[cep]} (current {salience_raw[j]*100:.1f}% raw)",
            min_value=-10,
            max_value=25,
            value=0,
            step=1,
        )

    # ----------------------------
    # Run simulation (WEIGHTED model layer)
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

    # KPIs (scaled to households)
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
        "Note: Percentages shown in sliders/table are **raw (unweighted)** to match your dashboard. "
        "All scenario math (reach + HH impact) is computed using **weights + deduplication**."
    )

    # ----------------------------
    # Diagnostics table
    # ----------------------------
    df = pd.DataFrame({
        "CEP Index": cep_idx,
        "CEP": [labels[i] for i in cep_idx],
        "CEP Type": [infer_cep_type(labels[i]) for i in cep_idx],

        # Dashboard-facing (RAW)
        "Category prevalence (raw %)": prevalence_raw * 100,
        "TFM salience (raw %)": salience_raw * 100,

        # Modeling (weighted) — kept for transparency but not required for the dashboard
        "Category prevalence (wtd %)": prevalence_w * 100,
        "TFM salience (wtd current %)": salience_w * 100,
        "TFM salience (wtd scenario %)": salience_target_w * 100,

        "Uplift (pts)": uplifts,

        # Household sizing uses WEIGHTED prevalence to scale to population HHs
        "Accessible TAM (HHs)": prevalence_w * hh_base,

        # Diagnostic only (not deduped)
        "Brand TAM current (HHs) [diagnostic]": (prevalence_w * salience_w) * hh_base,
        "Brand TAM scenario (HHs) [diagnostic]": (prevalence_w * salience_target_w) * hh_base,
        "Δ Brand TAM (HHs) [diagnostic]": ((prevalence_w * salience_target_w) - (prevalence_w * salience_w)) * hh_base,
    }).sort_values("Category prevalence (raw %)", ascending=False)

    st.subheader("CEP diagnostics")
    st.caption(
        "Brand TAM columns are **not deduplicated** (CEPs overlap by design). "
        "The KPI at top is **deduplicated Unique HH Reach**."
    )
    st.dataframe(df, use_container_width=True)

    # ----------------------------
    # Bubble matrix (RAW axes; WEIGHTED bubble sizing)
    # ----------------------------
    st.subheader("CEP Opportunity Bubble Matrix (Scenario)")
    st.caption(
        "Axes are **raw %** (dashboard-aligned). Bubble size is **Accessible TAM (HHs)** based on weighted prevalence × HH base."
    )

    chart_df = df.copy()

    bubble = (
        alt.Chart(chart_df)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("Category prevalence (raw %)", title="Category prevalence (raw %)"),
            y=alt.Y("TFM salience (raw %)", title="TFM salience (raw %)"),
            size=alt.Size(
                "Accessible TAM (HHs)",
                title="Accessible TAM (HHs)",
                scale=alt.Scale(range=[120, 3200])
            ),
            color=alt.Color("CEP Type", title="CEP Type"),
            tooltip=[
                "CEP",
                "CEP Type",
                alt.Tooltip("Category prevalence (raw %)", format=".1f"),
                alt.Tooltip("TFM salience (raw %)", format=".1f"),
                alt.Tooltip("Accessible TAM (HHs)", format=",.0f"),
                alt.Tooltip("Δ Brand TAM (HHs) [diagnostic]", format=",.0f"),
            ],
        )
        .properties(height=520)
    )

    x_med = float(chart_df["Category prevalence (raw %)"].median())
    y_med = float(chart_df["TFM salience (raw %)"].median())
    vline = alt.Chart(pd.DataFrame({"x": [x_med]})).mark_rule(strokeDash=[4, 4]).encode(x="x")
    hline = alt.Chart(pd.DataFrame({"y": [y_med]})).mark_rule(strokeDash=[4, 4]).encode(y="y")

    st.altair_chart(bubble + vline + hline, use_container_width=True)
    st.caption(f"Dotted lines are medians (raw): x={x_med:.1f}%, y={y_med:.1f}%.")


if __name__ == "__main__":
    main()



