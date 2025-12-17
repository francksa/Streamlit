#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 23:07:36 2025

@author: franck
"""

import streamlit as st
import pandas as pd
import altair as alt

# -----------------------------
# 1. Core assumptions & data
# -----------------------------

TFM_HOUSEHOLDS = 70_132_819  # households in TFM 22-state footprint

CEP_DATA = [
    {
        "cep": "Weekly grocery shopping",
        "segment": "Routine",
        "category_prevalence": 0.58,
        "tfm_salience_current": 0.08,
    },
    {
        "cep": "Trying to save money",
        "segment": "Routine",
        "category_prevalence": 0.50,
        "tfm_salience_current": 0.07,
    },
    {
        "cep": "Planning meals / low on supplies",
        "segment": "Routine",
        "category_prevalence": 0.36,
        "tfm_salience_current": 0.10,
    },
    {
        "cep": "Errands + groceries",
        "segment": "Routine",
        "category_prevalence": 0.34,
        "tfm_salience_current": 0.09,
    },
    {
        "cep": "Inspired to cook",
        "segment": "Inspiration",
        "category_prevalence": 0.28,
        "tfm_salience_current": 0.21,
    },
    {
        "cep": "Healthy / better-quality options",
        "segment": "Health",
        "category_prevalence": 0.26,
        "tfm_salience_current": 0.36,
    },
    {
        "cep": "Avoiding crowds / long lines",
        "segment": "Convenience",
        "category_prevalence": 0.25,
        "tfm_salience_current": 0.15,
    },
    {
        "cep": "Groceries online / pickup",
        "segment": "Convenience",
        "category_prevalence": 0.24,
        "tfm_salience_current": 0.09,
    },
    {
        "cep": "Ready-to-eat meals",
        "segment": "Convenience",
        "category_prevalence": 0.23,
        "tfm_salience_current": 0.14,
    },
    {
        "cep": "Health-conscious eating",
        "segment": "Health",
        "category_prevalence": 0.20,
        "tfm_salience_current": 0.33,
    },
    {
        "cep": "Short on time (convenient food)",
        "segment": "Convenience",
        "category_prevalence": 0.19,
        "tfm_salience_current": 0.14,
    },
    {
        "cep": "Try new or seasonal products",
        "segment": "Inspiration",
        "category_prevalence": 0.19,
        "tfm_salience_current": 0.20,
    },
    {
        "cep": "Hosting guests / special meal",
        "segment": "Inspiration",
        "category_prevalence": 0.16,
        "tfm_salience_current": 0.17,
    },
    {
        "cep": "Large family or group",
        "segment": "Specialty",
        "category_prevalence": 0.15,
        "tfm_salience_current": 0.10,
    },
    {
        "cep": "Specialty / international foods",
        "segment": "Specialty",
        "category_prevalence": 0.11,
        "tfm_salience_current": 0.23,
    },
    {
        "cep": "Need help / ideas on what to buy",
        "segment": "Specialty",
        "category_prevalence": 0.10,
        "tfm_salience_current": 0.15,
    },
    {
        "cep": "Environmentally friendly options",
        "segment": "Health",
        "category_prevalence": 0.10,
        "tfm_salience_current": 0.27,
    },
]


def build_base_df() -> pd.DataFrame:
    df = pd.DataFrame(CEP_DATA)
    df["accessible_tam_hh"] = df["category_prevalence"] * TFM_HOUSEHOLDS
    df["brand_tam_current_hh"] = (
        df["accessible_tam_hh"] * df["tfm_salience_current"]
    )
    return df


def run_simulation(
    salience_uplifts_pts: dict,
    trips_per_hh: float,
    spend_per_trip: float,
    conversion_rate: float,
) -> pd.DataFrame:
    """
    Run CEP simulation given uplift (in percentage points) by CEP
    and simple commercial assumptions.
    """
    df = build_base_df()

    # Scenario salience
    df["tfm_salience_scenario"] = df["tfm_salience_current"]
    for cep, uplift_pts in salience_uplifts_pts.items():
        mask = df["cep"] == cep
        uplift_prop = uplift_pts / 100.0
        df.loc[mask, "tfm_salience_scenario"] = (
            df.loc[mask, "tfm_salience_current"] + uplift_prop
        ).clip(upper=0.90)  # cap at 90% to avoid impossible values

    # Scenario Brand TAM
    df["brand_tam_scenario_hh"] = (
        df["accessible_tam_hh"] * df["tfm_salience_scenario"]
    )
    df["delta_brand_tam_hh"] = (
        df["brand_tam_scenario_hh"] - df["brand_tam_current_hh"]
    )

    # Commercial layer
    df["trips_per_hh"] = trips_per_hh
    df["spend_per_trip"] = spend_per_trip
    df["conversion_rate"] = conversion_rate

    df["revenue_current"] = (
        df["brand_tam_current_hh"]
        * df["trips_per_hh"]
        * df["spend_per_trip"]
        * df["conversion_rate"]
    )
    df["revenue_scenario"] = (
        df["brand_tam_scenario_hh"]
        * df["trips_per_hh"]
        * df["spend_per_trip"]
        * df["conversion_rate"]
    )
    df["delta_revenue"] = df["revenue_scenario"] - df["revenue_current"]

    return df


# -----------------------------
# 2. Streamlit UI
# -----------------------------

def main():
    st.set_page_config(
        page_title="TFM CEP TAM Simulator",
        layout="wide",
    )

    st.title("The Fresh Market – CEP TAM Simulator")
    st.markdown(
        "Interactively explore how changes in **CEP salience** "
        "could expand Brand TAM (households) and revenue."
    )

    base_df = build_base_df()

    # --- Sidebar controls ---
    st.sidebar.header("Simulation controls")

    st.sidebar.subheader("Commercial assumptions")
    trips = st.sidebar.number_input(
        "Trips per household per year (per CEP)",
        min_value=1.0,
        max_value=20.0,
        value=4.0,
        step=0.5,
    )
    spend = st.sidebar.number_input(
        "Average spend per trip ($)",
        min_value=5.0,
        max_value=200.0,
        value=40.0,
        step=1.0,
    )
    conv = st.sidebar.slider(
        "Conversion: share of 'TFM comes to mind' HHs that choose TFM",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.05,
    )

    st.sidebar.subheader("Salience uplifts (percentage points)")
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

    st.sidebar.markdown(
        "*Tip: leave uplift at 0 for CEPs you are not actively targeting.*"
    )

    # --- Run simulation ---
    sim_df = run_simulation(
        salience_uplifts_pts=salience_uplifts,
        trips_per_hh=trips,
        spend_per_trip=spend,
        conversion_rate=conv,
    )

    # --- KPIs ---
    total_brand_tam_current = sim_df["brand_tam_current_hh"].sum()
    total_brand_tam_scenario = sim_df["brand_tam_scenario_hh"].sum()
    total_delta_brand_tam = sim_df["delta_brand_tam_hh"].sum()

    total_rev_current = sim_df["revenue_current"].sum()
    total_rev_scenario = sim_df["revenue_scenario"].sum()
    total_delta_rev = sim_df["delta_revenue"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Brand TAM – current (HHs)",
        f"{total_brand_tam_current:,.0f}",
    )
    col2.metric(
        "Brand TAM – scenario (HHs)",
        f"{total_brand_tam_scenario:,.0f}",
        f"{total_delta_brand_tam:,.0f}",
    )
    col3.metric(
        "Annual revenue – scenario vs current",
        f"${total_rev_scenario:,.0f}",
        f"${total_delta_rev:,.0f}",
    )

    # --- Detailed table ---
    st.subheader("CEP-level results")

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
        "revenue_current",
        "revenue_scenario",
        "delta_revenue",
    ]

    table_df = sim_df[display_cols].copy()
    # Format percentages as % for display
    table_df["category_prevalence"] = table_df["category_prevalence"] * 100
    table_df["tfm_salience_current"] = table_df["tfm_salience_current"] * 100
    table_df["tfm_salience_scenario"] = table_df["tfm_salience_scenario"] * 100

    st.dataframe(
        table_df.rename(
            columns={
                "cep": "CEP",
                "segment": "CEP Type",
                "category_prevalence": "Category prevalence (%)",
                "accessible_tam_hh": "Accessible TAM (HHs)",
                "tfm_salience_current": "TFM salience – current (%)",
                "tfm_salience_scenario": "TFM salience – scenario (%)",
                "brand_tam_current_hh": "Brand TAM – current (HHs)",
                "brand_tam_scenario_hh": "Brand TAM – scenario (HHs)",
                "delta_brand_tam_hh": "Δ Brand TAM (HHs)",
                "revenue_current": "Revenue – current ($)",
                "revenue_scenario": "Revenue – scenario ($)",
                "delta_revenue": "Δ Revenue ($)",
            }
        ),
        use_container_width=True,
    )

    # --- Bubble matrix (scenario) ---
    st.subheader("CEP Opportunity Bubble Matrix – Scenario")

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
                title="Category Accessible TAM (prevalence, %)",
            ),
            y=alt.Y(
                "tfm_salience_scenario_pct",
                title="TFM Brand Fit (scenario salience, %)",
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
                alt.Tooltip("category_prevalence_pct", format=".1f"),
                alt.Tooltip("tfm_salience_current", title="Salience current", format=".2f"),
                alt.Tooltip("tfm_salience_scenario", title="Salience scenario", format=".2f"),
                alt.Tooltip("accessible_tam_hh", format=",.0f"),
                alt.Tooltip("brand_tam_current_hh", format=",.0f"),
                alt.Tooltip("brand_tam_scenario_hh", format=",.0f"),
                alt.Tooltip("delta_brand_tam_hh", format=",.0f"),
                alt.Tooltip("delta_revenue", format=",.0f"),
            ],
        )
        .properties(height=500)
    )

    # Add median lines for rough quadrants
    vline = alt.Chart(pd.DataFrame({"x": [26]})).mark_rule(strokeDash=[4, 4]).encode(
        x="x"
    )
    hline = alt.Chart(pd.DataFrame({"y": [15]})).mark_rule(strokeDash=[4, 4]).encode(
        y="y"
    )

    st.altair_chart(bubble + vline + hline, use_container_width=True)

    st.caption(
        "Bubble size = Accessible TAM (HHs). Vertical/horizontal dashed lines "
        "are rough medians to indicate high/low TAM and high/low brand fit."
    )


if __name__ == "__main__":
    main()
