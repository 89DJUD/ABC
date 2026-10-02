"""Fonctions Streamlit partagées entre les pages : chargement, paramètres, calculs mis en cache."""
from pathlib import Path

import pandas as pd
import streamlit as st

import abc_core as core

DEFAULT_CSV = Path(__file__).parent / "inventory_data.csv"
DEFAULTS = {"w_" + c: 1.0 for c in core.CRITERIA} | {"seuils": (20, 50)}


@st.cache_data(max_entries=5)
def load_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


@st.cache_data(max_entries=20)
def cached_topsis(df, weights: tuple, pct_a, pct_ab):
    return core.run_topsis(df, dict(zip(core.CRITERIA, weights)), pct_a, pct_ab)


@st.cache_resource(max_entries=5, show_spinner="Entraînement des 8 modèles…")
def cached_training(_df, classes: tuple, data_key: str):
    return core.train_and_evaluate(_df, list(classes))


def init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


def reset_settings():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def sidebar():
    df = load_csv(DEFAULT_CSV)
    with st.sidebar:
        st.subheader("Poids des critères TOPSIS", divider="gray")
        for c in core.CRITERIA:
            st.slider(c, 0.0, 5.0, step=0.5, key="w_" + c)
        weights = tuple(st.session_state["w_" + c] for c in core.CRITERIA)
        if sum(weights) == 0:
            st.error("Au moins un poids doit être positif.", icon=":material/error:")
            st.stop()
        st.caption("Poids normalisés : " + " · ".join(
            f"{c} {w / sum(weights):.2f}" for c, w in zip(core.CRITERIA, weights)))

        st.subheader("Seuils ABC (% d'articles)", divider="gray")
        pct_a, pct_ab = st.slider("Limites A | B | C", 1, 99, key="seuils",
                                  help="Part cumulée d'articles : A jusqu'à la 1re borne, B jusqu'à la 2e.")
        st.caption(f"A = {pct_a} % · B = {pct_ab - pct_a} % · C = {100 - pct_ab} %")
        st.button("Réinitialiser", icon=":material/restart_alt:", on_click=reset_settings, width="stretch")

    return df, weights, pct_a, pct_ab
