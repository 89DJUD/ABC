import streamlit as st

import abc_ui

st.set_page_config(page_title="Analyse ABC multicritère", page_icon=":material/inventory_2:", layout="wide")
abc_ui.init_state()

page = st.navigation([
    st.Page("app_pages/donnees.py", title="Données", icon=":material/table_chart:", default=True),
    st.Page("app_pages/topsis_abc.py", title="TOPSIS et ABC", icon=":material/leaderboard:"),
    st.Page("app_pages/machine_learning.py", title="Machine learning", icon=":material/model_training:"),
    st.Page("app_pages/approche_floue.py", title="Approche floue", icon=":material/blur_on:"),
])

df, weights, pct_a, pct_ab = abc_ui.sidebar()
st.session_state["ctx"] = {
    "df": df, "weights": weights, "pct_a": pct_a, "pct_ab": pct_ab,
    "res": abc_ui.cached_topsis(df, weights, pct_a, pct_ab),
}
page.run()
