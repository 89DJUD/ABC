import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr

import abc_core as core
import abc_ui

ctx = st.session_state["ctx"]
df, res = ctx["df"], ctx["res"]
pct_a, pct_ab = ctx["pct_a"], ctx["pct_ab"]

st.title("Approche floue : Fuzzy TOPSIS et classes ABC floues")
st.caption("Attributs en nombres flous triangulaires, poids linguistiques, distance des sommets (Chen, 2000), "
           "puis appartenance graduelle aux classes A, B, C.")

with st.container(border=True):
    st.subheader("Paramètres flous")
    c1, c2 = st.columns(2)
    spread = c1.slider("Incertitude des mesures quantitatives (± %)", 0, 30, 10) / 100
    delta = c2.slider("Demi-largeur de la zone de transition entre classes (points de centile)", 0, 15, 5) / 100
    cols = st.columns(len(core.CRITERIA))
    terms = tuple(col.selectbox(f"Poids {c}", list(core.FUZZY_WEIGHT_TERMS), index=2, key=f"fw_{c}")
                  for col, c in zip(cols, core.CRITERIA))

cc_f = abc_ui.cached_fuzzy(df, spread, terms)
rank_f, classe_f = core.abc_from_scores(cc_f, pct_a, pct_ab)
p = (rank_f - 0.5) / len(df)
mA, mB, mC = core.fuzzy_memberships(p, pct_a, pct_ab, delta)

fz = df.copy()
fz["CC net"], fz["Classe nette"] = res["Score TOPSIS"], res["Classe"]
fz["CC flou"], fz["Rang flou"], fz["Classe floue"] = cc_f, rank_f, classe_f
fz["µA"], fz["µB"], fz["µC"] = mA, mB, mC
fz["Statut"] = np.where(fz[["µA", "µB", "µC"]].max(axis=1) < 1, "Frontière", "Net")

with st.container(horizontal=True):
    st.metric("Spearman CC net / CC flou", f"{spearmanr(fz['CC net'], fz['CC flou'])[0]:.4f}", border=True)
    st.metric("Classes inchangées", f"{(fz['Classe nette'] == fz['Classe floue']).mean():.1%}", border=True)
    st.metric("Changements A ↔ C", int(((fz["Classe nette"] == "A") & (fz["Classe floue"] == "C")).sum()
                                      + ((fz["Classe nette"] == "C") & (fz["Classe floue"] == "A")).sum()), border=True)
    st.metric("Articles en zone frontière", int((fz["Statut"] == "Frontière").sum()), border=True)

color = alt.Color("Classe nette:N", scale=alt.Scale(domain=core.LABELS, range=list(core.CLASS_COLORS.values())))
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("TOPSIS net vs Fuzzy TOPSIS")
        st.altair_chart(alt.Chart(fz.reset_index(names="Article")).mark_circle(size=25, opacity=0.7).encode(
            x=alt.X("CC net:Q", scale=alt.Scale(zero=False)), y=alt.Y("CC flou:Q", scale=alt.Scale(zero=False)),
            color=color, tooltip=["Article", "Classe nette", "Classe floue",
                                  alt.Tooltip("CC net:Q", format=".4f"), alt.Tooltip("CC flou:Q", format=".4f")],
        ).properties(height=320))
with right:
    with st.container(border=True):
        st.subheader("Partition floue A / B / C")
        g = np.linspace(0, 1, 401)
        curves = pd.DataFrame({"Rang centile (%)": np.tile(g * 100, 3),
                               "Appartenance": np.concatenate(core.fuzzy_memberships(g, pct_a, pct_ab, delta)),
                               "Classe": np.repeat(core.LABELS, len(g))})
        st.altair_chart(alt.Chart(curves).mark_line(strokeWidth=2.5).encode(
            x="Rang centile (%):Q", y="Appartenance:Q",
            color=alt.Color("Classe:N", scale=alt.Scale(domain=core.LABELS, range=list(core.CLASS_COLORS.values()))),
        ).properties(height=320))

with st.container(border=True):
    st.subheader("Matrice de passage des classes")
    st.dataframe(pd.crosstab(fz["Classe nette"], fz["Classe floue"], rownames=["TOPSIS net"], colnames=["Fuzzy TOPSIS"]))

with st.container(border=True):
    st.subheader("Articles")
    only_border = st.toggle("Uniquement les articles en zone frontière", value=True)
    view = fz[fz["Statut"] == "Frontière"] if only_border else fz
    pc = st.column_config.ProgressColumn
    st.dataframe(
        view.sort_values("Rang flou")[["Rang flou", "Classe floue", "µA", "µB", "µC", "CC flou",
                                       "Classe nette", "CC net"] + core.COLUMNS],
        height=420,
        column_config={k: pc(k, format="%.2f", min_value=0.0, max_value=1.0) for k in ["µA", "µB", "µC"]},
    )
    st.download_button("Télécharger les résultats flous", fz.to_csv(index_label="Article").encode("utf-8"),
                       "inventory_data_classe_floue.csv", "text/csv", icon=":material/download:")
