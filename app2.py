import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# CONFIG — update these filenames if needed
# ============================================================
RAW_DATA_PATH = "2510110_research_science_raw_data (1).csv"

# Either of these codebooks can supply CEP labels; we’ll try both.
LEVELS_CODEBOOK_PATH = "2510110_research_science_levels_codebook.csv"
QUESTION_CODEBOOK_PATH = "2510110_research_science_question_codebook - 2510110_research_science_question_codebook.csv"

# Household base used to scale proportions -> households (keep your 2023 base unless you have MSA HH base)
HOUSEHOLD_BASE_DEFAULT = 70_132_819

# Monte Carlo defaults
DEFAULT_N_SIMS = 800
DEFAULT_CAP = 0.90

# ============================================================
# Helpers
# ============================================================
def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.sum(w))


def to_binary_selected(series: pd.Series) -> np.ndarray:
    """
    Convert 1=Selected, 2=Not Selected (and NaN -> Not Selected) to 0/1.
    """
    s = series.fillna(2)
    return (s.astype(float) == 1.0).astype(int).to_numpy()


@st.cache_data(show_spinner=False)
def load_data():
    raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)

    levels_cb = None
    try:
        levels_cb = pd.read_csv(LEVELS_CODEBOOK_PATH, low_memory=False)
    except Exception:
        pass

    question_cb = None
    try:
        question_cb = pd.read_csv(QUESTION_CODEBOOK_PATH, low_memory=False)
    except Exception:
        pass

    return raw, levels_cb, question_cb


def discover_ceps(raw: pd.DataFrame):
    """
    Find CEP indices i where BOTH:
      - RS1_{i}NET exists (category CEP prevalence)
      - RS8_{i}_1NET exists (TFM association for CEP i; brand 1 = TFM)

    This is the corrected CEP-first convention.
    """
    cep_idx = []
    for col in raw.columns:
        m = re.match(r"RS1_(\d+)NET$", col)
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
    Best-effort CEP label extraction.

    Priority:
      1) Levels codebook (often has RS1_{i}NET code==1 value as item text)
      2) Question codebook (varies; we attempt a few patterns)
      3) Fallback: "CEP {i}"
    """
    labels = {i: f"CEP {i}" for i in cep_idx}

    # 1) Try Levels codebook: question == RS1_{i}NET and code == 1 -> 'value'
    if levels_cb is not None:
        cb = levels_cb.copy()
        cb.columns = [c.strip().lower() for c in cb.columns]
        # We expect columns like: question, code, value
        if {"question", "code", "value"}.issubset(set(cb.columns)):
            for i in cep_idx:
                q = f"RS1_{i}NET"
                hit = cb[(cb["question"].astype(str) == q) & (cb["code"].astype(str) == "1")]
                if len(hit) > 0:
                    labels[i] = str(hit.iloc[0]["value"]).strip()

    # 2) Try Question codebook if labels still generic
    if question_cb is not None:
        qc = question_cb.copy()
        qc.columns = [c.strip().lower() for c in qc.columns]

        # We’ll attempt to use any columns that look like "qid"/"question"/"text"/"label"
        # This code is intentionally defensive.
        possible_id_cols = [c for c in qc.columns if c in ("qid", "question", "variable", "var")]
        possible_text_cols = [c for c in qc.columns if c in ("text", "label", "questiontext", "question_text", "description")]

        if possible_id_cols and possible_text_cols:
            id_col = possible_id_cols[0]
            text_col = possible_text_cols[0]

            for i in cep_idx:
                if labels[i].startswith("CEP "):  # still fallback
                    # try RS1_iNET label
                    hit = qc[qc[id_col].astype(str).str.upper() == f"RS1_{i}NET".upper()]
                    if len(hit) > 0:
                        candidate = str(hit.iloc[0][text_col]).strip()
                        if candidate and candidate.lower() != "nan":
                            labels[i] = candidate

    return labels


def infer_cep_type(label: str) -> str:
    """
    Optional heuristic categorization for coloring bubbles.
    Edit freely if you want different logic.
    """
    s = label.lower()

    routine_kw = ["weekly", "regular", "save money", "low on supplies", "errands"]
    health_kw = ["healthy", "health-conscious", "eat better", "better-quality", "environment"]
    convenience_kw = ["short on time", "convenient", "ready-to-eat", "pickup", "online", "crowded", "long lines"]
    inspiration_kw = ["inspired", "try something new", "new or seasonal", "seasonal", "hosting", "special meal"]
    specialty_kw = ["specialty", "international", "ideas", "help", "large family", "group"]

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
    salience_current: np.ndarray,
    uplifts_pts: np.ndarray,
    n_sims: int,
    cap: float,
    seed: int = 7,
):
    """
    Respondent-level Monte Carlo with weights + true deduplication.

    X: n_resp x k binary matrix (current TFM selected per CEP)
    salience_current: weighted baseline salience per CEP (length k)
    uplifts_pts: desired uplift in percentage points per CEP (length k)

    We convert uplift into a target salience s' and flip non-selectors with probability:
        flip_prob = (s' - s) / (1 - s)
    so expected salience reaches target (in expectation).
    """
    rng = np.random.default_rng(seed)

    uplift = uplifts_pts / 100.0
    s = salience_current
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
This simulator uses **respondent-level data** and **weights (`wts`)** to compute a true, deduplicated measure:

- **Unique HH Reach (TFM associated with ≥1 CEP)**

It applies salience uplifts at the **CEP level**, simulates household-level uptake using **Monte Carlo** (so overlap is handled correctly),
and recomputes **deduped** reach. Multiple CEP uplifts naturally show **diminishing returns**.
"""
    )

    raw, levels_cb, question_cb = load_data()

    # Validate required columns
    if "wts" not in raw.columns:
        st.error("Missing required column: wts")
        return

    w = raw["wts"].to_numpy()

    # Discover CEPs (correct naming: RS8_{cep}_1NET)
    cep_idx = discover_ceps(raw)
    labels = get_cep_labels(levels_cb, question_cb, cep_idx)
    k = len(cep_idx)

    st.caption(f"Detected **{k} CEPs** with both prevalence and TFM association data.")

    # Build prevalence + association matrices
    prev_cols = [f"RS1_{i}NET" for i in cep_idx]
    tfm_cols = [f"RS8_{i}_1NET" for i in cep_idx]

    prev = np.column_stack([to_binary_selected(raw[c]) for c in prev_cols])  # 0/1
    X = np.column_stack([to_binary_selected(raw[c]) for c in tfm_cols])      # 0/1

    # Baselines
    # Baseline deduped reach: prefer RS8mpen_1 if present, else compute from X
    if "RS8mpen_1" in raw.columns:
        mpen = to_binary_selected(raw["RS8mpen_1"])
        unique_reach_current = wmean(mpen, w)
        # Also compute from X to validate
        mpen_from_x = (X.max(axis=1) > 0).astype(int)
        unique_reach_from_x = wmean(mpen_from_x, w)
    else:
        mpen_from_x = (X.max(axis=1) > 0).astype(int)
        unique_reach_current = wmean(mpen_from_x, w)
        unique_reach_from_x = unique_reach_current

    salience_current = np.array([wmean(X[:, j], w) for j in range(k)])
    prevalence = np.array([wmean(prev[:, j], w) for j in range(k)])

    # Sidebar controls
    st.sidebar.header("Simulation controls")
    hh_base = st.sidebar.number_input("Household base (for scaling to HHs)", value=int(HOUSEHOLD_BASE_DEFAULT), step=100000)
    n_sims = st.sidebar.slider("Monte Carlo runs", 200, 3000, DEFAULT_N_SIMS, step=100)
    cap = st.sidebar.slider("Salience cap", 0.50, 0.95, DEFAULT_CAP, step=0.01)

    st.sidebar.subheader("TFM salience uplifts (percentage points)")
    uplifts = np.zeros(k, dtype=float)

    # Sort display by category prevalence (descending) so top CEPs are easier to find
    order = np.argsort(-prevalence)
    ordered_idx = [cep_idx[i] for i in order]

    for display_pos, cep in enumerate(ordered_idx):
        j = cep_idx.index(cep)  # map to matrix column index
        uplifts[j] = st.sidebar.slider(
            f"{labels[cep]} (current {salience_current[j]*100:.1f}%)",
            min_value=-10,
            max_value=25,
            value=0,
            step=1,
        )

    # Run simulation
    unique_reach_scenario, reach_dist, salience_target = simulate_unique_reach(
        X=X,
        w=w,
        salience_current=salience_current,
        uplifts_pts=uplifts,
        n_sims=n_sims,
        cap=cap,
        seed=7
    )

    # KPIs
    unique_hh_current = unique_reach_current * hh_base
    unique_hh_scenario = unique_reach_scenario * hh_base
    delta_unique_hh = unique_hh_scenario - unique_hh_current

    lo = float(np.quantile(reach_dist, 0.10)) * hh_base
    hi = float(np.quantile(reach_dist, 0.90)) * hh_base

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique HH Reach (current, deduped)", f"{unique_hh_current:,.0f}")
    c2.metric("Unique HH Reach (scenario, deduped)", f"{unique_hh_scenario:,.0f}", f"{delta_unique_hh:,.0f}")
    c3.metric("Scenario range (P10–P90)", f"{lo:,.0f} – {hi:,.0f}")

    # Validation metric vs RS8mpen_1 (if present)
    diff_pp = (unique_reach_from_x - unique_reach_current) * 100
    c4.metric("Dedup validation (X vs RS8mpen_1)", f"{diff_pp:+.2f} pp")

    st.caption(
        "Validation is shown as **(dedup from RS8 grid) minus (RS8mpen_1)** in percentage points. "
        "Ideally this is ~0.00 pp."
    )

    # Diagnostics table
    df = pd.DataFrame({
        "CEP Index": cep_idx,
        "CEP": [labels[i] for i in cep_idx],
        "CEP Type": [infer_cep_type(labels[i]) for i in cep_idx],
        "Category prevalence (%)": prevalence * 100,
        "TFM salience current (%)": salience_current * 100,
        "TFM salience scenario (%)": salience_target * 100,
        "Uplift (pts)": uplifts,
        "Accessible TAM (HHs)": prevalence * hh_base,
        # Diagnostic only (not deduped)
        "Brand TAM current (HHs) [diagnostic]": (prevalence * salience_current) * hh_base,
        "Brand TAM scenario (HHs) [diagnostic]": (prevalence * salience_target) * hh_base,
        "Δ Brand TAM (HHs) [diagnostic]": ((prevalence * salience_target) - (prevalence * salience_current)) * hh_base,
    })

    df = df.sort_values("Category prevalence (%)", ascending=False)

    st.subheader("CEP diagnostics (weighted)")
    st.caption(
        "Note: Brand TAM columns are **not deduplicated** (CEPs overlap by design). "
        "Your headline KPI is the **deduplicated Unique HH Reach** above."
    )
    st.dataframe(df, use_container_width=True)

    # Bubble matrix
    st.subheader("CEP Opportunity Bubble Matrix (Scenario)")
    st.caption("X = Category prevalence (%), Y = TFM salience scenario (%), Bubble size = Accessible TAM (HHs)")

    chart_df = df.copy()

    bubble = (
        alt.Chart(chart_df)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("Category prevalence (%)", title="Category prevalence (%)"),
            y=alt.Y("TFM salience scenario (%)", title="TFM salience (scenario, %)"),
            size=alt.Size("Accessible TAM (HHs)", title="Accessible TAM (HHs)",
                          scale=alt.Scale(range=[120, 3200])),
            color=alt.Color("CEP Type", title="CEP Type"),
            tooltip=[
                "CEP",
                "CEP Type",
                alt.Tooltip("Category prevalence (%)", format=".1f"),
                alt.Tooltip("TFM salience current (%)", format=".1f"),
                alt.Tooltip("TFM salience scenario (%)", format=".1f"),
                alt.Tooltip("Accessible TAM (HHs)", format=",.0f"),
                alt.Tooltip("Δ Brand TAM (HHs) [diagnostic]", format=",.0f"),
            ],
        )
        .properties(height=520)
    )

    # Dynamic median lines (data-driven)
    x_med = float(chart_df["Category prevalence (%)"].median())
    y_med = float(chart_df["TFM salience scenario (%)"].median())
    vline = alt.Chart(pd.DataFrame({"x": [x_med]})).mark_rule(strokeDash=[4, 4]).encode(x="x")
    hline = alt.Chart(pd.DataFrame({"y": [y_med]})).mark_rule(strokeDash=[4, 4]).encode(y="y")

    st.altair_chart(bubble + vline + hline, use_container_width=True)
    st.caption(f"Dotted lines are medians (dynamic): x={x_med:.1f}%, y={y_med:.1f}%.")


if __name__ == "__main__":
    main()

