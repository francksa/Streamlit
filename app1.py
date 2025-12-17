#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 13:04:18 2025

@author: franck
"""

import streamlit as st
import pandas as pd
import altair as alt

# -----------------------------
# 1. Core assumptions & data
# -----------------------------

# Households in TFM footprint (22 states) – replace with MSA-only HHs if you have them
TFM_HOUSEHOLDS = 70_132_819

# CEP data using **MSA-only** category & TFM salience percentages
# All percentages expressed as proportions (0–1)
CEP_DATA = [
    {
        "cep": "Weekly grocery shopping",
        "segment": "Routine",
        "category_prevalence": 0.60,  # 60%
        "tfm_salience_current": 0.08,  # 8%
    },
    {
        "cep": "Trying to save money",
        "segment": "Routine",
        "category_prevalence": 0.48,
        "tfm_salience_current": 0.08,
    },
    {
        "cep": "Planning meals / low on supplies",
        "segment": "Routine",
        "category_prevalence": 0.35,
        "tfm_salience_current": 0.09,
    },
    {
        "cep": "Errands + groceries",
        "segment": "Routine",
        "category_prevalence": 0.30,
        "tfm_salience_current": 0.10,
    },
    {
        "cep": "Inspired to cook",
        "segment": "Inspiration",
        "category_prevalence": 0.27,
        "tfm_salience_current": 0.23,
    },
    {
        "cep": "Healthy / better-quality options",
        "segment": "Health",
        "category_prevalence": 0.27,
        "tfm_salience_current": 0.34,
    },
    {
        "cep": "Avoiding crowds / long lines",
        "segment": "Convenience",
        "category_prevalence": 0.24,
        "tfm_salience_current": 0.14,
    },
    {
        "cep": "Ready-to-eat meals",
        "segment": "Convenience",
        "category_prevalence": 0.24,
        "tfm_salience_current": 0.16,
    },
    {
        "cep": "Groceries online / pickup",
        "segment": "Convenience",
        "category_prevalence": 0.23,
        "tfm_salience_current": 0.11,
    },
    {
        "cep": "Try new or seasonal products",
        "segment": "Inspiration",
        "category_prevalence": 0.19,
        "tfm_salience_current": 0.22,
    },
    {
        "cep": "Health-conscious eating",
        "segment": "Health",
        "category_prevalence": 0.19,
        "tfm_salience_current": 0.34,
    },
    {
        "cep": "Short on time (convenient food)",
        "segment": "Convenience",
        "category_prevalence": 0.18,
        "tfm_salience_current": 0.17,
    },
    {
        "cep": "Hosting guests / special meal",
        "segment": "Inspiration",
        "category_prevalence": 0.17,
        "tfm_salience_current": 0.19,
    },
    {
        "cep": "Large family or group",
        "segment": "Specialty",
        "category_prevalence": 0.16,
        "tfm_salience_current": 0.09,
    },
    {
        "cep": "Specialty / international foods",
        "segment": "Specialty",
        "category_prevalence": 0.11,
        "tfm_salience_current": 0.24,
    },
    {
        "cep": "Environmentally friendly options",
        "segment": "Health",
        "category_prevalence": 0.09,
        "tfm_salience_current": 0.24,
    },
    {
        "cep": "Need help / ideas on what to buy",
        "segment": "Specialty",
        "category_prevalence": 0.07,
        "tfm_salience_current": 0.16,
    },
]


def build_base_df() -> pd.DataFrame:
    df = pd.DataFrame(CEP_DATA)
    df["accessible_tam_hh"] = df["category_prevalence"] * TFM_HOUSEHOLDS
    df["brand_tam_current_hh"] = (
        df["accessible_tam_hh"] * df["tfm_salience_current"]
    )
    return df


def run_simulation(salience_uplifts_pts: dict) -> pd.DataFrame:
    """
    Run CEP simulation given uplift (in percentage points) by CEP.
    No revenue / commercial layer – purely Mental Availability and TAM.
    """
    df = build_base_df()

    # Scenario salience
    df["tfm_salience_scenario"] = df["tfm_salience_current"]
    for cep, uplift_pts in salience_uplifts_pts.items():
        mask = df["cep"] == cep
        uplift_prop = uplift_pts / 100.0
        df.loc[mask, "tfm_salience_scenario"] = (
            df.loc[mask, "tfm_salience_current"] + uplift_prop
        ).clip(upper=0.90)  # simple cap to avoid impossible values

    # Scenario Brand TAM (households)
    df["brand_tam_scenario_hh"] = (
        df["accessible_tam_hh"] * df["tfm_salience_scenario"]
    )
    df["delta_brand_tam_hh"] = (
        df["brand_tam_scenario_hh"] - df["brand_tam_current_hh"]
    )

    return df


# -----------------------------
# 2. Streamlit UI
# -----------------------------

def main():
    st.set_page_config(
        page_title="TFM CEP TAM Simulator (MSA-only)",
        layout="wide",
    )

    st.title("The Fresh Market – CEP TAM Simulator (MSA Footprint)")
    st.markdown(
        "This tool models how changes in **CEP salience** within the MSAs where TFM "
        "has stores impact **Brand TAM (households)**."
    )

    base_df = build_base_df()

    # --- Sidebar: salience uplifts only ---
    st.sidebar.header("Salience uplift controls")
    st.sidebar.markdown(
        "Adjust **percentage point lifts** in TFM salience per CEP.\n\n"
        "Leave at 0 for CEPs you’re not actively targeting."
    )

    salience_uplifts = {}
    for _, row in base_df.iterrows():
        cep = row["cep"]
        current_pct = row["tfm_salience_current"] * 100
        uplift = st.sidebar.slider(
            f"{cep}  (current: {current_pct:.1f}%)",
            min_value=-10,
            max_value=20,
            value=0,
            step=1,
        )
        if uplift != 0:
            salience_uplifts[cep] = uplift

    # --- Run simulation ---
    sim_df = run_simulation(salience_uplifts_pts=salience_uplifts)

    # --- KPIs ---
    total_brand_tam_current = sim_df["brand_tam_current_hh"].sum()
    total_brand_tam_scenario = sim_df["brand_tam_scenario_hh"].sum()
    total_delta_brand_tam = sim_df["delta_brand_tam_hh"].sum()

    col1, col2 = st.columns(2)
    col1.metric(
        "Brand TAM – current (HHs)",
        f"{total_brand_tam_current:,.0f}",
    )
    col2.metric(
        "Brand TAM – scenario (HHs)",
        f"{total_brand_tam_scenario:,.0f}",
        f"{total_delta_brand_tam:,.0f}",
    )

    # --- Detailed table ---
    st.subheader("CEP-level results (MSA-only)")

    display_cols = [
        "cep",
        "segment",
        "category_prevalence",
        "accessible_tam_hh",
        "tfm_salience_current",
        "tfm_salience_scenario",
        "brand_tam_current_hh",
        "brand_tam_scenario_hh",
        "delta_brand_tam_hh",
    ]

    table_df = sim_df[display_cols].copy()
    # Format percentages as %
    table_df["category_prevalence"] = table_df["category_prevalence"] * 100
    table_df["tfm_salience_current"] = table_df["tfm_salience_current"] * 100
    table_df["tfm_salience_scenario"] = table_df["tfm_salience_scenario"] * 100

    st.dataframe(
        table_df.rename(
            columns={
                "cep": "CEP",
                "segment": "CEP Type",
                "category_prevalence": "Category prevalence in MSAs (%)",
                "accessible_tam_hh": "Accessible TAM in MSAs (HHs)",
                "tfm_salience_current": "TFM salience – current (%)",
                "tfm_salience_scenario": "TFM salience – scenario (%)",
                "brand_tam_current_hh": "Brand TAM – current (HHs)",
                "brand_tam_scenario_hh": "Brand TAM – scenario (HHs)",
                "delta_brand_tam_hh": "Δ Brand TAM (HHs)",
            }
        ),
        use_container_width=True,
    )

    # --- Bubble matrix (scenario) ---
    st.subheader("CEP Opportunity Bubble Matrix – Scenario (MSA-only)")

    chart_df = sim_df.copy()
    chart_df["category_prevalence_pct"] = chart_df["category_prevalence"] * 100
    chart_df["tfm_salience_scenario_pct"] = (
        chart_df["tfm_salience_scenario"] * 100
    )

    bubble = (
        alt.Chart(chart_df)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X(
                "category_prevalence_pct",
                title="Category CEP prevalence in MSA footprint (%)",
            ),
            y=alt.Y(
                "tfm_salience_scenario_pct",
                title="TFM salience (scenario, %)",
            ),
            size=alt.Size(
                "accessible_tam_hh",
                title="Accessible TAM (HHs)",
                scale=alt.Scale(range=[100, 3000]),
            ),
            color=alt.Color("segment", title="CEP Type"),
            tooltip=[
                "cep",
                "segment",
                alt.Tooltip("category_prevalence_pct", format=".1f",
                            title="Category prevalence (%)"),
                alt.Tooltip("tfm_salience_current", format=".2f",
                            title="TFM salience current (prop)"),
                alt.Tooltip("tfm_salience_scenario", format=".2f",
                            title="TFM salience scenario (prop)"),
                alt.Tooltip("accessible_tam_hh", format=",.0f",
                            title="Accessible TAM (HHs)"),
                alt.Tooltip("brand_tam_current_hh", format=",.0f",
                            title="Brand TAM current (HHs)"),
                alt.Tooltip("brand_tam_scenario_hh", format=",.0f",
                            title="Brand TAM scenario (HHs)"),
                alt.Tooltip("delta_brand_tam_hh", format=",.0f",
                            title="Δ Brand TAM (HHs)"),
            ],
        )
        .properties(height=500)
    )

    # Rough quadrant medians (you can tweak these)
    vline = alt.Chart(pd.DataFrame({"x": [25]})).mark_rule(strokeDash=[4, 4]).encode(
        x="x"
    )
    hline = alt.Chart(pd.DataFrame({"y": [15]})).mark_rule(strokeDash=[4, 4]).encode(
        y="y"
    )

    st.altair_chart(bubble + vline + hline, use_container_width=True)

    st.caption(
        "Bubble size = Accessible TAM (households) within MSAs. "
        "Dashed lines approximate high/low TAM and high/low brand fit."
    )


if __name__ == "__main__":
    main()
