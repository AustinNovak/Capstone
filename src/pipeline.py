"""Shared data, preprocessing and metric helpers (same recipe as the Milestone 4 revision notebook)."""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from m4_utils import IQRWinsorizer

RND = 42
ORIG7 = ["LAT_DD", "LON_DD", "month", "lcond", "PHSTVL", "state", "ECOREGL3"]
EXPANDED = ORIG7 + ["YEAR", "Temp", "LWSKM2", "NO3", "alk", "HUC"]
CAT_COLS = ["state", "ECOREGL3", "Dataset", "HUC"]
GEO_NUM = ["LAT_DD", "LON_DD", "month", "YEAR"]


def load(path="data/cleaned_dataset_expanded.csv", sample_sites=None):
    df = pd.read_csv(path, dtype={"SITE_ID": str, "ECOREGL3": str, "HUC": str})
    if sample_sites:  # small deterministic subset of sites, used by CI
        keep = np.random.RandomState(RND).choice(df.SITE_ID.unique(), sample_sites, replace=False)
        df = df[df.SITE_ID.isin(keep)].reset_index(drop=True)
    return df


def split(df):
    """Held-out TEST sites = fold 0 of a 5-fold stratified, site-grouped split (never used for tuning)."""
    le = LabelEncoder(); y = le.fit_transform(df["GENUS"])
    te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=RND).split(df, y, df["SITE_ID"])))[1]
    dev = np.setdiff1d(np.arange(len(df)), te)
    return le, y, dev, te


def build_pre(feats):
    geo = [c for c in feats if c in GEO_NUM]; cat = [c for c in feats if c in CAT_COLS]
    chem = [c for c in feats if c not in geo + cat]
    parts = [("geo", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), geo),
             ("chem", Pipeline([("w", IQRWinsorizer(k=1.5)), ("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), chem),
             ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False), cat)]
    return ColumnTransformer([p for p in parts if p[2]])


def full_proba(model, X, k):
    P = model.predict_proba(X); out = np.zeros((len(X), k)); out[:, model.classes_] = P; return out


def metrics(model, X, y, k):
    full = full_proba(model, X, k); pred = full.argmax(1)
    top3 = float((np.argsort(full, 1)[:, -3:] == np.asarray(y)[:, None]).any(1).mean())
    return dict(macro_f1=f1_score(y, pred, average="macro", zero_division=0), accuracy=accuracy_score(y, pred), top3=top3), pred
