"""Validation checks run by CI (GitHub Actions) on every push."""
import os, sys, pickle, json
import numpy as np
sys.path.insert(0, "src")
from pipeline import load, split, build_pre, metrics, ORIG7, EXPANDED
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_required_files_exist():
    for f in ["README.md", "requirements.txt", "src/train_mlflow.py", "src/pipeline.py", "src/m4_utils.py", "data/cleaned_dataset.csv",
              "data/cleaned_dataset_expanded.csv", ".github/workflows/ci.yml", "notebooks/milestone4_notebook_revised.ipynb", "outputs/results/final_test_results.csv"]:
        assert os.path.exists(os.path.join(ROOT, f)), f"missing {f}"


def test_data_columns_and_split_by_site():
    df = load(sample_sites=200)
    assert set(EXPANDED + ["GENUS", "SITE_ID"]).issubset(df.columns)
    le, y, dev, te = split(df)
    assert not set(df.SITE_ID.iloc[dev]) & set(df.SITE_ID.iloc[te]), "test sites leaked into the development set"


def test_pipeline_trains_and_predicts():
    df = load(sample_sites=200); le, y, dev, te = split(df)
    pipe = Pipeline([("pre", build_pre(EXPANDED)), ("model", RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0))]).fit(df.iloc[dev][EXPANDED], y[dev])
    m, _ = metrics(pipe, df.iloc[te][EXPANDED], y[te], len(le.classes_))
    assert 0 <= m["top3"] <= 1 and m["top3"] > 0.05 and m["accuracy"] >= 0


def test_saved_model_loads_and_predicts():
    import m4_utils  # noqa: F401  (needed to unpickle the outlier-capping step)
    info = json.load(open(os.path.join(ROOT, "models/feature_lists.json")))
    model = pickle.load(open(os.path.join(ROOT, "models/tuned_expanded_logistic.pkl"), "rb"))
    df = load(sample_sites=50)
    assert len(model.predict(df[info["expanded"]].head(20))) == 20
