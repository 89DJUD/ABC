import altair as alt
import pandas as pd
import streamlit as st

import abc_core as core
import abc_ui

ctx = st.session_state["ctx"]
df, res = ctx["df"], ctx["res"]

st.title("Classification automatique par machine learning")
st.caption("Les classes ABC issues de TOPSIS servent de labels. Entrées : les 8 attributs bruts "
           "(qualitatives codées par la Table 1, standardisation). Découpage stratifié 70 % / 30 %.")

if res["Classe"].value_counts().reindex(core.LABELS, fill_value=0).min() < 10:
    st.warning("Chaque classe doit contenir au moins 10 articles pour entraîner les modèles. "
               "Ajustez les seuils ABC dans la barre latérale.", icon=":material/warning:")
    st.stop()

data_key = str(pd.util.hash_pandas_object(df).sum())
models, perf, cms, n_train, n_test = abc_ui.cached_training(df, tuple(res["Classe"]), data_key)
best = perf.index[0]

with st.container(horizontal=True):
    st.metric("Meilleur modèle (F1 macro)", best, border=True)
    st.metric("Accuracy", f"{perf.loc[best, 'Accuracy']:.1%}", border=True)
    st.metric("F1 macro", f"{perf.loc[best, 'F1 macro']:.3f}", border=True)
    st.metric("Rappel classe A", f"{perf.loc[best, 'Rappel A']:.1%}", border=True)
    st.metric("Apprentissage / test", f"{n_train} / {n_test}", border=True)

with st.container(border=True):
    st.subheader("Comparaison des modèles (jeu de test)")
    st.dataframe(perf.style.format("{:.3f}").background_gradient(cmap="Greens", axis=0))

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Indicateurs par modèle")
        metric = st.segmented_control("Indicateur", ["Accuracy", "F1 macro", "Rappel A", "Kappa", "AUC OvR"],
                                      default="F1 macro", label_visibility="collapsed")
        metric = metric or "F1 macro"
        st.altair_chart(alt.Chart(perf.reset_index()).mark_bar().encode(
            x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Modèle:N", sort="-x", title=None),
            tooltip=["Modèle", alt.Tooltip(f"{metric}:Q", format=".3f")],
        ).properties(height=300))
with right:
    with st.container(border=True):
        st.subheader("Matrice de confusion")
        name = st.selectbox("Modèle", perf.index, label_visibility="collapsed")
        cm = pd.DataFrame(cms[name], index=core.LABELS, columns=core.LABELS)
        cm = cm.rename_axis("Réelle").reset_index().melt("Réelle", var_name="Prédite", value_name="Articles")
        base = alt.Chart(cm).encode(x=alt.X("Prédite:N"), y=alt.Y("Réelle:N"))
        st.altair_chart((base.mark_rect().encode(color=alt.Color("Articles:Q", scale=alt.Scale(scheme="blues"), legend=None))
                         + base.mark_text(fontSize=16).encode(
                             text="Articles:Q",
                             color=alt.condition(alt.datum.Articles > cm["Articles"].max() / 2,
                                                 alt.value("white"), alt.value("black"))))
                        .properties(height=300))

st.subheader("Classer un nouvel article")
with st.form("new_item"):
    c1, c2, c3, c4 = st.columns(4)
    item = {
        "Risk": c1.selectbox("Risque", list(core.TABLE1["Risk"])),
        "Demand fluctuation": c2.selectbox("Fluctuation de la demande", list(core.TABLE1["Demand fluctuation"])),
        "Consignment stock": c3.selectbox("Stock en consignation", list(core.TABLE1["Consignment stock"])),
        "Unit size": c4.selectbox("Taille de l'unité", list(core.TABLE1["Unit size"])),
        "Average stock": c1.number_input("Stock moyen", 0.0, value=float(df["Average stock"].median())),
        "Daily usage": c2.number_input("Utilisation quotidienne", 0.0, value=float(df["Daily usage"].median())),
        "Unit cost": c3.number_input("Coût unitaire", 0.0, value=float(df["Unit cost"].median())),
        "Lead time": c4.number_input("Délai de livraison (jours)", 0, value=int(df["Lead time"].median())),
    }
    submitted = st.form_submit_button("Classer", icon=":material/category:", type="primary")

if submitted:
    new = pd.DataFrame([item])[core.COLUMNS]
    probas = pd.DataFrame(
        [dict(zip(models[m].classes_, models[m].predict_proba(core.features(new))[0])) | {"Modèle": m}
         for m in perf.index]).set_index("Modèle")[core.LABELS]
    probas.insert(0, "Classe prédite", probas.idxmax(axis=1))

    full = pd.concat([df[core.COLUMNS], new], ignore_index=True)
    ref = core.run_topsis(full, dict(zip(core.CRITERIA, ctx["weights"])), ctx["pct_a"], ctx["pct_ab"]).iloc[-1]

    with st.container(horizontal=True):
        st.metric(f"Prédiction — {best}", probas.loc[best, "Classe prédite"], border=True)
        st.metric("Référence TOPSIS (recalcul complet)", ref["Classe"],
                  f"CC = {ref['Score TOPSIS']:.4f} · rang {int(ref['Rang'])}/{len(full)}", delta_color="off", border=True)
        votes = probas["Classe prédite"].value_counts()
        st.metric("Vote des 8 modèles", votes.index[0], f"{votes.iloc[0]}/8 modèles", delta_color="off", border=True)
    st.dataframe(probas, column_config={k: st.column_config.ProgressColumn(k, format="%.2f", min_value=0.0, max_value=1.0)
                                        for k in core.LABELS})
