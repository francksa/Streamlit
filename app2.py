import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# 1) FILE & UNIVERSE CONFIGURATION (same as your working script)
# ============================================================
RAW_DATA_FILE = "TFM_RAW_MSA.csv"
DASHBOARD_EXPORT = "Category Entry Points (CEPs).csv"

HH_UNIVERSE = 70_132_819  # households in TFM footprint (22 states) or your chosen HH base

WEIGHT_COL = "wts"
AWARE_COL = "RS3_1NET"  # brand awareness / eligibility base

DEFAULT_N_SIMS = 800
DEFAULT_CAP = 0.95
SEED = 7

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

# Optional grouping for colors
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
# 2) DATA UTILITIES (same spirit as your working script)
# ============================================================
def clean_numeric(val):
    """Handles percentages, decimals, and string formatting."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        f = float(val)
        # if dashboard exported 48 (meaning 48%), convert
        return f / 100.0 if f > 1.0 else f
    s = str(val).strip().replace(",", "")
    has_pct = "%" in s
    s = s.replace("%", "")
    try:
        f = float(s)
        if has_pct or f > 1.0:
            return f / 100.0
        return f
    except ValueError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_and_align_data():
    if not Path(DASHBOARD_EXPORT).exists() or not Path(RAW_DATA_FILE).exists():
        raise FileNotFoundError(
            f"Missing files. App needs `{DASHBOARD_EXPORT}` and `{RAW_DATA_FILE}` in the same folder as app.py."
        )

    # Load with flexible encoding
    try:
        dash_df = pd.read_csv(DASHBOARD_EXPORT, encoding="utf-8-sig")
        raw_df = pd.read_csv(RAW_DATA_FILE, encoding="utf-8-sig", low_memory=False)
    except Exception:
        dash_df = pd.read_csv(DASHBOARD_EXPORT, encoding="latin1")
        raw_df = pd.read_csv(RAW_DATA_FILE, encoding="latin1", low_memory=False)

    raw_df[WEIGHT_COL] = pd.to_numeric(raw_df[WEIGHT_COL], errors="coerce").fillna(0.0)

    # Identify split point in dashboard export (category vs brand section)
    text_col = dash_df.iloc[:, 0].astype(str)
    brand_start = text_col.str.contains("Your Brand's CEP Salience", case=False, na=False).idxmax()

    cat_section = dash_df.iloc[:brand_start].copy()
    brand_section = dash_df.iloc[brand_start:].copy()

    baselines = []
    for cid, label in CEP_NAME_MAP.items():
        # prevalence from cat section
        cat_match = cat_section[cat_section.iloc[:, 0].astype(str).str.contains(label, na=False, case=False)]
        # salience from brand section
        brand_match = brand_section[brand_section.iloc[:, 0].astype(str).str.contains(label, na=False, case=False)]

        if not cat_match.empty and not brand_match.empty:
            baselines.append(
                {
                    "id": cid,
                    "label": label,
                    "segment": infer_cep_type(label),
                    "prevalence": clean_numeric(cat_match.iloc[0, 1]),
                    "salience": clean_numeric(brand_match.iloc[0, 1]),
                }
            )

    cep_df = pd.DataFrame(baselines)
    if cep_df.empty or len(cep_df) < 10:
        raise ValueError(
            "Could not extract CEP baselines from the dashboard export. "
            "Check that the CEP labels in CEP_NAME_MAP match the export text exactly."
        )

    # Ensure stable ordering by CEP id
    cep_df = cep_df.sort_values("id").reset_index(drop=True)
    return cep_df, raw_df


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    denom = np.sum(w)
    if denom <= 0:
        return 0.0
    return float(np.sum(x * w) / denom)


def asked_mask(series: pd.Series) -> np.ndarray:
    """True where RS8 has a coded response (1/2)."""
    return (series.notna() & series.isin([1, 2])).to_numpy()


# ============================================================
# 3) MONTE CARLO DEDUP ENGINE (upgraded)
# ============================================================
def run_hh_simulation(raw_df: pd.DataFrame, cep_df: pd.DataFrame, uplifts_pts: np.ndarray,
                      n_sims: int = DEFAULT_N_SIMS, cap: float = DEFAULT_CAP, seed: int = SEED,
                      require_rs8_answered: bool = True):
    """
    - Baseline salience/prevalence from dashboard export (cep_df)
    - Monte Carlo dedup based on respondent RS8 associations in raw_df
    - Eligibility for flips: AWARE base (RS3_1NET==1) AND (optionally) RS8 answered (1/2)
    """
    rng = np.random.default_rng(seed)

    wts = raw_df[WEIGHT_COL].to_numpy()
    aware = (raw_df[AWARE_COL] == 1).to_numpy()

    n = len(raw_df)
    k = len(cep_df)

    # Association matrix X from RS8_{cid}_1NET (1=selected)
    X = np.zeros((n, k), dtype=int)
    rs8_answered = np.ones((n, k), dtype=bool)

    for j, cid in enumerate(cep_df["id"].tolist()):
        col = f"RS8_{cid}_1NET"
        if col not in raw_df.columns:
            # if missing, keep zeros and set not answered
            rs8_answered[:, j] = False
            continue

        s = raw_df[col]
        m = asked_mask(s)
        rs8_answered[:, j] = m
        # treat missing as 0 in X, but control flips via eligibility
        X[m, j] = (s[m] == 1).astype(int).to_numpy()

    # Eligibility: aware AND (optionally) answered that RS8 item
    if require_rs8_answered:
        elig = rs8_answered & aware[:, None]
    else:
        elig = aware[:, None]

    # Baseline salience from dashboard export (vector)
    curr_s = cep_df["salience"].to_numpy(dtype=float)

    # Target salience after uplift
    target_s = np.minimum(curr_s + (uplifts_pts / 100.0), cap)

    # Flip probabilities to move expected mean from curr_s to target_s
    denom = (1.0 - curr_s)
    need = (target_s - curr_s)
    flip_probs = np.where(denom > 1e-9, need / denom, 0.0)
    flip_probs = np.clip(flip_probs, 0.0, 1.0)

    # Current deduped reach from observed X (weighted)
    curr_mpen = wmean((X.max(axis=1) > 0).astype(int), wts)

    # Monte Carlo distribution
    sim_dist = np.zeros(n_sims, dtype=float)
    for t in range(n_sims):
        X_sim = X.copy()
        U = rng.random((n, k))
        flips = (X_sim == 0) & elig & (U < flip_probs)
        X_sim[flips] = 1
        sim_dist[t] = wmean((X_sim.max(axis=1) > 0).astype(int), wts)

    scen_mpen = float(np.mean(sim_dist))
    return curr_mpen, scen_mpen, sim_dist, target_s


# ============================================================
# 4) STREAMLIT UI
# ============================================================
def main():
    st.set_page_config(layout="wide", page_title="TFM Mental Availability Simulator (Deduped)")
    st.title("TFM Household Growth & Mental Availability Simulator (Deduped)")

    st.markdown(
        """
This simulator is **anchored to your dashboard export** for baseline metrics:
- Category prevalence (CEP salience)  
- TFM salience by CEP  

Then it uses the respondent-level **MSA raw file** to simulate **deduped unique household reach** under salience uplifts.
"""
    )

    # Load data
    try:
        cep_df, raw_df = load_and_align_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Sidebar controls
    st.sidebar.header("Model Controls")
    hh_base = st.sidebar.number_input("Household universe (HHs)", value=int(HH_UNIVERSE), step=100000)

    n_sims = st.sidebar.slider("Monte Carlo runs", 200, 3000, DEFAULT_N_SIMS, step=100)
    cap = st.sidebar.slider("Max salience cap", 0.70, 0.99, DEFAULT_CAP, step=0.01)

    require_rs8_answered = st.sidebar.checkbox(
        "Require RS8 answered (recommended)",
        value=True,
        help="Prevents flips for CEPs where a respondent has no coded RS8 response (NaN). Keeps simulation consistent with what was measured."
    )

    # Sliders ordered by prevalence (dashboard style)
    st.sidebar.header("TFM Salience Uplift (pts)")
    sorted_ceps = cep_df.sort_values("prevalence", ascending=False).reset_index(drop=True)

    uplifts_by_id = {}
    for _, row in sorted_ceps.iterrows():
        base_pct = row["salience"] * 100
        uplift = st.sidebar.slider(
            f"{row['label']} (Base: {base_pct:.1f}%)",
            min_value=-10,
            max_value=30,
            value=0,
            step=1,
            key=f"s_{int(row['id'])}"
        )
        uplifts_by_id[int(row["id"])] = uplift

    # Build uplift array aligned to cep_df id order
    uplifts_pts = np.array([uplifts_by_id[int(i)] for i in cep_df["id"].tolist()], dtype=float)

    # Run simulation
    curr_mpen, scen_mpen, sim_dist, target_sal = run_hh_simulation(
        raw_df=raw_df,
        cep_df=cep_df,
        uplifts_pts=uplifts_pts,
        n_sims=n_sims,
        cap=cap,
        seed=SEED,
        require_rs8_answered=require_rs8_answered
    )

    # HH impact
    curr_hhs = curr_mpen * hh_base
    scen_hhs = scen_mpen * hh_base
    new_hhs = scen_hhs - curr_hhs

    p10 = float(np.quantile(sim_dist, 0.10)) * hh_base
    p90 = float(np.quantile(sim_dist, 0.90)) * hh_base

    # KPI row
    st.subheader("Simulated Market Impact (Deduped Unique HH Reach)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Unique HHs", f"{curr_hhs:,.0f}")
    m2.metric("Scenario Unique HHs", f"{scen_hhs:,.0f}", f"{new_hhs:,.0f}")
    m3.metric("Scenario band (P10–P90)", f"{p10:,.0f} – {p90:,.0f}")
    m4.metric("Current reach (weighted)", f"{curr_mpen*100:.1f}%")

    # Build display df (dashboard baselines + scenario)
    out = cep_df.copy()
    out["uplift_pts"] = uplifts_pts
    out["salience_scenario"] = target_sal
    out["accessible_tam_hh"] = out["prevalence"] * hh_base

    # (Diagnostics) Non-deduped “Brand TAM” view is optional; keep off by default
    show_diagnostics = st.toggle("Show non-deduped diagnostic TAM columns", value=False)
    if show_diagnostics:
        out["brand_tam_current_hh (diag)"] = out["accessible_tam_hh"] * out["salience"]
        out["brand_tam_scenario_hh (diag)"] = out["accessible_tam_hh"] * out["salience_scenario"]
        out["delta_brand_tam_hh (diag)"] = out["brand_tam_scenario_hh (diag)"] - out["brand_tam_current_hh (diag)"]

    # Table
    st.subheader("CEP Metrics (Dashboard Baseline + Scenario)")
    table = out.copy()
    table["prevalence_%"] = table["prevalence"] * 100
    table["salience_%"] = table["salience"] * 100
    table["salience_scenario_%"] = table["salience_scenario"] * 100

    cols = [
        "label", "segment",
        "prevalence_%", "accessible_tam_hh",
        "salience_%", "uplift_pts", "salience_scenario_%"
    ]
    if show_diagnostics:
        cols += ["brand_tam_current_hh (diag)", "brand_tam_scenario_hh (diag)", "delta_brand_tam_hh (diag)"]

    table_disp = table[cols].rename(columns={
        "label": "CEP",
        "segment": "CEP Type",
        "prevalence_%": "Category prevalence (%)",
        "accessible_tam_hh": "Accessible TAM (HHs)",
        "salience_%": "TFM salience – current (%)",
        "uplift_pts": "Uplift (pts)",
        "salience_scenario_%": "TFM salience – scenario (%)",
    }).sort_values("Category prevalence (%)", ascending=False)

    st.dataframe(table_disp, use_container_width=True)

    # Download
    csv_bytes = table_disp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download table as CSV",
        data=csv_bytes,
        file_name="tfm_cep_simulator_output.csv",
        mime="text/csv"
    )

    # Bubble matrix (pretty)
    st.subheader("CEP Opportunity Bubble Matrix (Scenario)")
    st.caption("X = Category prevalence (%). Y = TFM salience scenario (%). Bubble size = Accessible TAM (HHs).")

    chart_df = table_disp.copy()

    bubble = (
        alt.Chart(chart_df)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("Category prevalence (%)", title="Category prevalence (%)"),
            y=alt.Y("TFM salience – scenario (%)", title="TFM salience – scenario (%)"),
            size=alt.Size("Accessible TAM (HHs)", title="Accessible TAM (HHs)", scale=alt.Scale(range=[120, 3200])),
            color=alt.Color("CEP Type", title="CEP Type"),
            tooltip=[
                "CEP",
                "CEP Type",
                alt.Tooltip("Category prevalence (%)", format=".1f"),
                alt.Tooltip("TFM salience – current (%)", format=".1f"),
                alt.Tooltip("TFM salience – scenario (%)", format=".1f"),
                alt.Tooltip("Accessible TAM (HHs)", format=",.0f"),
                alt.Tooltip("Uplift (pts)", format=".0f"),
            ],
        )
        .properties(height=520)
    )

    x_med = float(chart_df["Category prevalence (%)"].median())
    y_med = float(chart_df["TFM salience – scenario (%)"].median())

    vline = alt.Chart(pd.DataFrame({"x": [x_med]})).mark_rule(strokeDash=[4, 4]).encode(x="x")
    hline = alt.Chart(pd.DataFrame({"y": [y_med]})).mark_rule(strokeDash=[4, 4]).encode(y="y")

    st.altair_chart(bubble + vline + hline, use_container_width=True)
    st.caption(f"Dotted lines are medians (dynamic): x={x_med:.1f}%, y={y_med:.1f}%.")

    # Optional: distribution view
    with st.expander("Show Monte Carlo distribution (unique reach)"):
        dist_df = pd.DataFrame({"Unique Reach (weighted %)": sim_dist * 100})
        hist = alt.Chart(dist_df).mark_bar().encode(
            x=alt.X("Unique Reach (weighted %):Q", bin=alt.Bin(maxbins=25)),
            y="count()"
        ).properties(height=200)
        st.altair_chart(hist, use_container_width=True)


if __name__ == "__main__":
    main()

