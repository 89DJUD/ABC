import altair as alt
import pandas as pd
import streamlit as st

import abc_core as core

df = st.session_state["ctx"]["df"]

st.title("Analyse ABC multicritère des stocks")
st.caption("Codification (Table 1) → critères agrégés (Table 2) → TOPSIS → classes ABC → "
           "machine learning. Les paramètres se règlent dans la barre latérale.")

with st.container(horizontal=True):
    st.metric("Articles", len(df), border=True)
    st.metric("Variables", len(core.COLUMNS), border=True)
    st.metric("Qualitatives", len(core.QUALI), border=True)
    st.metric("Quantitatives", len(core.QUANTI), border=True)
    st.metric("Valeurs manquantes", int(df[core.COLUMNS].isna().sum().sum()), border=True)

nature = {
    "Risk": "Qualitative ordinale", "Demand fluctuation": "Qualitative ordinale",
    "Average stock": "Quantitative continue", "Daily usage": "Quantitative continue",
    "Unit cost": "Quantitative continue", "Lead time": "Quantitative discrète",
    "Consignment stock": "Qualitative binaire", "Unit size": "Qualitative ordinale",
}
rows = []
for c in core.COLUMNS:
    if c in core.TABLE1:
        detail = " · ".join(f"{k} → {v:.2f}" for k, v in core.TABLE1[c].items())
    else:
        detail = f"[{df[c].min():.2f} ; {df[c].max():.2f}], moyenne {df[c].mean():.2f}"
    rows.append({"Variable": c, "Nature": nature[c], "Modalités (score Table 1) ou plage": detail})

with st.container(border=True):
    st.subheader("Nature des variables")
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "Variable": st.column_config.TextColumn(width="small"),
        "Nature": st.column_config.TextColumn(width="small"),
        "Modalités (score Table 1) ou plage": st.column_config.TextColumn(width="large")})

left, right = st.columns([2, 3])
with left:
    with st.container(border=True):
        st.subheader("Distribution")
        var = st.selectbox("Variable", core.COLUMNS, label_visibility="collapsed")
        if var in core.TABLE1:
            counts = df[var].value_counts().reindex(list(core.TABLE1[var])).reset_index()
            chart = alt.Chart(counts).mark_bar(color=core.PRIMARY, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X(f"{var}:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("count:Q", title="Articles"),
                tooltip=[var, "count"])
        else:
            chart = alt.Chart(df).mark_bar(color=core.PRIMARY, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X(f"{var}:Q", bin=alt.Bin(maxbins=25)), y=alt.Y("count()", title="Articles"))
        st.altair_chart(chart.properties(height=300))
with right:
    with st.container(border=True):
        st.subheader("Données brutes")
        st.dataframe(df.rename_axis("Article"), height=370)
