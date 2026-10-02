"""Logique métier partagée : scores Table 1, critères Table 2, TOPSIS, ABC, Fuzzy TOPSIS, ML."""
import numpy as np
import pandas as pd

QUALI = ["Risk", "Demand fluctuation", "Consignment stock", "Unit size"]
QUANTI = ["Average stock", "Daily usage", "Unit cost", "Lead time"]
COLUMNS = ["Risk", "Demand fluctuation", "Average stock", "Daily usage",
           "Unit cost", "Lead time", "Consignment stock", "Unit size"]

TABLE1 = {
    "Risk":               {"High": 0.47, "Normal": 0.35, "Low": 0.18},
    "Demand fluctuation": {"Increasing": 0.36, "Stable": 0.28, "Unknown": 0.20, "Decreasing": 0.16, "Ending": 0.00},
    "Consignment stock":  {"No": 0.80, "Yes": 0.20},
    "Unit size":          {"Large": 0.53, "Medium": 0.31, "Small": 0.13},
}
TABLE2 = {
    "Criticality": {"Risk": 0.78, "Demand fluctuation": 0.22},
    "Demand":      {"Daily usage": 0.71, "Average stock": 0.29},
    "Supply":      {"Lead time": 0.75, "Consignment stock": 0.25},
}
CRITERIA = ["Criticality", "Demand", "Supply", "Cost", "Size"]
LABELS = ["A", "B", "C"]
CLASS_COLORS = {"A": "#C44E52", "B": "#DD8452", "C": "#4C72B0"}

TFN_LING = {
    "Risk": {"Low": (0.18, 0.18, 0.35), "Normal": (0.18, 0.35, 0.47), "High": (0.35, 0.47, 0.47)},
    "Demand fluctuation": {"Ending": (0.00, 0.00, 0.16), "Decreasing": (0.00, 0.16, 0.20),
                           "Unknown": (0.00, 0.20, 0.36), "Stable": (0.20, 0.28, 0.36),
                           "Increasing": (0.28, 0.36, 0.36)},
    "Consignment stock": {"No": (0.80, 0.80, 0.80), "Yes": (0.20, 0.20, 0.20)},
    "Unit size": {"Small": (0.13, 0.13, 0.31), "Medium": (0.13, 0.31, 0.53), "Large": (0.31, 0.53, 0.53)},
}
FUZZY_WEIGHT_TERMS = {
    "Peu important": (0.0, 0.25, 0.5),
    "Moyennement important": (0.25, 0.5, 0.75),
    "Important": (0.5, 0.75, 1.0),
    "Très important": (0.75, 1.0, 1.0),
}


def validate(df):
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return f"Colonnes manquantes : {', '.join(missing)}"
    for c, m in TABLE1.items():
        bad = set(df[c].unique()) - set(m)
        if bad:
            return f"Modalités inconnues pour {c} : {', '.join(map(str, bad))}"
    return None


def scores(df):
    """Scores Table 1 pour les qualitatives, min-max pour les quantitatives."""
    S = pd.DataFrame(index=df.index)
    for c in COLUMNS:
        if c in TABLE1:
            S[c] = df[c].map(TABLE1[c])
        else:
            rng = df[c].max() - df[c].min()
            S[c] = (df[c] - df[c].min()) / rng if rng else 0.0
    return S


def criteria(df):
    S = scores(df)
    X = pd.DataFrame(index=df.index)
    for g, comp in TABLE2.items():
        X[g] = sum(w * S[a] for a, w in comp.items())
    X["Cost"] = df["Unit cost"]
    X["Size"] = S["Unit size"]
    return X


def topsis(M, w):
    M = np.asarray(M, float)
    w = np.asarray(w, float) / np.sum(w)
    V = M / np.sqrt((M ** 2).sum(axis=0)) * w
    d_pos = np.sqrt(((V - V.max(0)) ** 2).sum(1))
    d_neg = np.sqrt(((V - V.min(0)) ** 2).sum(1))
    return d_neg / (d_pos + d_neg)


def abc_from_scores(cc, pct_a, pct_ab):
    """Classes ABC par quantiles de rang (score décroissant)."""
    n = len(cc)
    rank = pd.Series(cc).rank(ascending=False, method="first").astype(int).values
    n_a, n_ab = int(round(pct_a / 100 * n)), int(round(pct_ab / 100 * n))
    classe = np.where(rank <= n_a, "A", np.where(rank <= n_ab, "B", "C"))
    return rank, classe


def run_topsis(df, weights, pct_a, pct_ab):
    X = criteria(df)
    cc = topsis(X.values, [weights[c] for c in CRITERIA])
    rank, classe = abc_from_scores(cc, pct_a, pct_ab)
    out = df.copy()
    out[CRITERIA] = X
    out["Score TOPSIS"] = cc
    out["Rang"] = rank
    out["Classe"] = classe
    out["Valeur annuelle"] = df["Daily usage"] * 365 * df["Unit cost"]
    return out


def fuzzy_topsis(df, spread, weight_terms):
    def fuzzify(col):
        if col in TFN_LING:
            return np.array([TFN_LING[col][v] for v in df[col]])
        x = df[col].values.astype(float)
        T = np.c_[(1 - spread) * x, x, (1 + spread) * x]
        lo, hi = T[:, 0].min(), T[:, 2].max()
        return (T - lo) / (hi - lo)

    F = {c: fuzzify(c) for c in COLUMNS}
    FX = {g: sum(w * F[a] for a, w in comp.items()) for g, comp in TABLE2.items()}
    FX["Cost"], FX["Size"] = F["Unit cost"], F["Unit size"]

    n = len(df)
    d_pos, d_neg = np.zeros(n), np.zeros(n)
    for c in CRITERIA:
        w = np.array(FUZZY_WEIGHT_TERMS[weight_terms[c]])
        V = FX[c] / FX[c][:, 2].max() * w
        d_pos += np.sqrt(((V - w) ** 2).mean(1))
        d_neg += np.sqrt((V ** 2).mean(1))
    return d_neg / (d_pos + d_neg), FX


def fuzzy_memberships(p, pct_a, pct_ab, delta):
    """Partition floue (Ruspini) sur le rang centile p ∈ [0, 1]."""
    def ramp_down(x, a, b):
        return np.clip((b - x) / (b - a), 0, 1) if b > a else (x <= a).astype(float)
    a, ab = pct_a / 100, pct_ab / 100
    mA = ramp_down(p, a - delta, a + delta)
    mC = 1 - ramp_down(p, ab - delta, ab + delta)
    mB = np.clip(1 - mA - mC, 0, 1)
    return mA, mB, mC


# ---------------- Machine learning
def features(df):
    Fe = pd.DataFrame(index=df.index)
    for c in COLUMNS:
        Fe[c] = df[c].map(TABLE1[c]) if c in TABLE1 else df[c].astype(float)
    return Fe


def model_zoo(seed=42):
    """Modèles avec les hyperparamètres retenus par GridSearchCV dans le notebook."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB

    def pipe(m):
        return Pipeline([("scaler", StandardScaler()), ("clf", m)])

    return {
        "Réseau de neurones (MLP)": pipe(MLPClassifier((32,), alpha=0.1, max_iter=3000, random_state=seed)),
        "SVM (RBF)": pipe(SVC(C=100, gamma=0.01, probability=True, random_state=seed)),
        "Régression logistique": pipe(LogisticRegression(C=10, max_iter=5000)),
        "Gradient boosting": pipe(GradientBoostingClassifier(n_estimators=200, max_depth=2, random_state=seed)),
        "Forêt aléatoire": pipe(RandomForestClassifier(n_estimators=300, random_state=seed)),
        "k-NN": pipe(KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "Naïve Bayes": pipe(GaussianNB()),
        "Arbre de décision": pipe(DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=seed)),
    }


def train_and_evaluate(df, classes, test_size=0.3, seed=42):
    import warnings
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 cohen_kappa_score, roc_auc_score, confusion_matrix)
    warnings.filterwarnings("ignore", category=FutureWarning)

    X, y = features(df), pd.Series(classes, index=df.index)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    models, rows, cms = {}, [], {}
    for name, m in model_zoo(seed).items():
        m.fit(X_tr, y_tr)
        yp, proba = m.predict(X_te), m.predict_proba(X_te)
        models[name] = m
        cms[name] = confusion_matrix(y_te, yp, labels=LABELS)
        rows.append({
            "Modèle": name,
            "Accuracy": accuracy_score(y_te, yp),
            "Précision macro": precision_score(y_te, yp, average="macro", zero_division=0),
            "Rappel macro": recall_score(y_te, yp, average="macro", zero_division=0),
            "F1 macro": f1_score(y_te, yp, average="macro", zero_division=0),
            "Rappel A": recall_score(y_te, yp, labels=["A"], average=None, zero_division=0)[0],
            "Kappa": cohen_kappa_score(y_te, yp),
            "AUC OvR": roc_auc_score(y_te, proba, multi_class="ovr", labels=LABELS),
        })
    perf = pd.DataFrame(rows).set_index("Modèle").sort_values("F1 macro", ascending=False)
    return models, perf, cms, len(X_tr), len(X_te)
