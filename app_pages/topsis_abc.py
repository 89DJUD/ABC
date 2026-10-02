import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import abc_core as core

ctx = st.session_state["ctx"]
res = ctx["res"].sort_values("Rang")
n = len(res)

st.title("TOPSIS et classification ABC")
st.caption("Score = coefficient de proximité CC à la solution idéale (5 critères à maximiser). "
           "Classes affectées par quantiles du rang.")

color = alt.Color("Classe:N", scale=alt.Scale(domain=core.LABELS, range=list(core.CLASS_COLORS.values())))
summary = res.groupby("Classe").agg(Nb=("Rang", "size"), CC_min=("Score TOPSIS", "min"),
                                    Valeur=("Valeur annuelle", "sum")).reindex(core.LABELS)
summary["Valeur %"] = 100 * summary["Valeur"] / summary["Valeur"].sum()

summary = summary.fillna(0)
kpis = st.columns(5)
for col, k in zip(kpis, core.LABELS):
    col.metric(f"Classe {k} (articles)", int(summary.loc[k, "Nb"]),
               f"{summary.loc[k, 'Valeur %']:.1f} % de la valeur",
               help="Part de la valeur annuelle de consommation (usage × 365 × coût) captée par la classe",
               delta_color="off", delta_arrow="off", border=True)
for col, k in zip(kpis[3:], ["A", "B"]):
    col.metric(f"Seuil CC classe {k}", f"{summary.loc[k, 'CC_min']:.4f}" if summary.loc[k, "Nb"] else "—",
               "CC minimal", delta_color="off", delta_arrow="off",
               help=f"Plus petit coefficient de proximité de la classe {k}", border=True)

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Score TOPSIS par rang")
        st.altair_chart(alt.Chart(res).mark_bar(width=1.5).encode(
            x=alt.X("Rang:Q", scale=alt.Scale(domain=[1, n], nice=False)),
            y=alt.Y("Score TOPSIS:Q", title="Coefficient de proximité CC"), color=color,
            tooltip=["Rang", alt.Tooltip("Score TOPSIS:Q", format=".4f"), "Classe", "Risk", "Unit size"],
        ).properties(height=320))
with right:
    with st.container(border=True):
        st.subheader("Courbes de Pareto")
        topsis_curve = pd.DataFrame({
            "% articles": res["Rang"] / n * 100,
            "% valeur": res["Valeur annuelle"].cumsum() / res["Valeur annuelle"].sum() * 100,
            "Classement": "TOPSIS multicritère"})
        v = res["Valeur annuelle"].sort_values(ascending=False)
        classic = pd.DataFrame({"% articles": np.arange(1, n + 1) / n * 100,
                                "% valeur": v.cumsum().values / v.sum() * 100,
                                "Classement": "ABC classique (valeur)"})
        rules = alt.Chart(pd.DataFrame({"x": [ctx["pct_a"], ctx["pct_ab"]]})).mark_rule(color=core.GRID, strokeDash=[2, 3]).encode(x="x:Q")
        st.altair_chart((alt.Chart(pd.concat([topsis_curve, classic])).mark_line().encode(
            x=alt.X("% articles:Q", title="% cumulé d'articles", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("% valeur:Q", title="% cumulé de la valeur annuelle"),
            color=alt.Color("Classement:N", title=None, legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=["TOPSIS multicritère", "ABC classique (valeur)"],
                                            range=[core.PRIMARY, "#A39E93"])),
            strokeDash=alt.StrokeDash("Classement:N", legend=None,
                                      scale=alt.Scale(domain=["TOPSIS multicritère", "ABC classique (valeur)"],
                                                      range=[[1, 0], [5, 3]])),
            strokeWidth=alt.value(2.5))
            + rules).properties(height=320))

with st.container(border=True):
    st.subheader("Profil moyen des classes")
    prof = res.groupby("Classe")[core.CRITERIA + ["Score TOPSIS"]].mean()
    for c in ["Risk", "Consignment stock", "Unit size"]:
        top = {"Risk": "High", "Consignment stock": "No", "Unit size": "Large"}[c]
        prof[f"% {c} = {top}"] = res.groupby("Classe")[c].apply(lambda s, t=top: 100 * (s == t).mean())
    st.dataframe(prof.style.format("{:.3f}").format("{:.1f}", subset=[c for c in prof if c.startswith("%")]))

with st.container(border=True):
    st.subheader("Classement des articles")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        classes = st.pills("Classes", core.LABELS, selection_mode="multi", default=core.LABELS)
        risk = st.multiselect("Risque", list(core.TABLE1["Risk"]), default=list(core.TABLE1["Risk"]))
    view = res[res["Classe"].isin(classes or []) & res["Risk"].isin(risk)]
    st.dataframe(
        view[["Rang", "Classe", "Score TOPSIS"] + core.COLUMNS + core.CRITERIA].rename_axis("Article"),
        height=420,
        column_config={"Score TOPSIS": st.column_config.ProgressColumn(
            "Score TOPSIS", format="%.4f", min_value=0.0, max_value=1.0)},
    )
    out = ctx["df"].copy()
    out[["Score TOPSIS", "Rang", "Classe"]] = ctx["res"][["Score TOPSIS", "Rang", "Classe"]]
    st.download_button("Télécharger la base avec la variable Classe", out.to_csv(index=False).encode("utf-8"),
                       "inventory_data_classe.csv", "text/csv", icon=":material/download:")
