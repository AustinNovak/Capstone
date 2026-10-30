"""Train the capstone models and track every run with MLflow (parameters, metrics, artifacts, model).

Usage:  python src/train_mlflow.py                 # full data, 4 models x 2 feature sets = 8 runs
        python src/train_mlflow.py --sample 300    # quick run on 300 sites (used by CI)
On Databricks, MLflow tracking is built in; locally runs go to ./mlflow.db (view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
Hyper-parameters are the ones chosen by GridSearchCV in the Milestone 4 revision (outputs/results/selected_hyperparameters.csv).
"""
import argparse, os, sys, json, tempfile
import mlflow, mlflow.sklearn, pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load, split, build_pre, metrics, ORIG7, EXPANDED, RND

MODELS = {  # (estimator, tuned hyper-parameters from GridSearchCV, needs sample weights)
    "Logistic Regression": (LogisticRegression(max_iter=300, random_state=RND), {"C": 0.1, "class_weight": "balanced"}, False),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=RND, n_jobs=-1), {"max_depth": 20, "min_samples_leaf": 5, "class_weight": "balanced_subsample"}, False),
    "Gradient Boosting": (HistGradientBoostingClassifier(max_iter=25, random_state=RND), {"learning_rate": 0.15, "max_depth": 5, "class_weight": "balanced"}, False),
    "Neural Network": (MLPClassifier(max_iter=30, early_stopping=True, n_iter_no_change=4, random_state=RND), {"hidden_layer_sizes": (64,), "alpha": 0.01}, True),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", type=int, default=None); ap.add_argument("--experiment", default="/Shared/fish-genus-capstone")
    ap.add_argument("--data", default="data/cleaned_dataset_expanded.csv"); a = ap.parse_args()
    if os.environ.get("DATABRICKS_RUNTIME_VERSION") is None: mlflow.set_tracking_uri("sqlite:///mlflow.db"); a.experiment = "fish-genus-capstone"
    mlflow.set_experiment(a.experiment)
    df = load(a.data, a.sample); le, y, dev, te = split(df); K = len(le.classes_)
    print(f"{len(df):,} rows, {df.SITE_ID.nunique():,} sites, {K} genera | dev {len(dev):,} test {len(te):,}")
    results = []
    for fs_name, feats in [("original7", ORIG7), ("expanded13", EXPANDED)]:
        for name, (est, params, sw) in MODELS.items():
            with mlflow.start_run(run_name=f"{name} | {fs_name}"):
                est = est.set_params(**params); pipe = Pipeline([("pre", build_pre(feats)), ("model", est)])
                kw = {"model__sample_weight": compute_sample_weight("balanced", y[dev])} if sw else {}
                pipe.fit(df.iloc[dev][feats], y[dev], **kw)
                m, pred = metrics(pipe, df.iloc[te][feats], y[te], K)
                mlflow.log_params({"model": name, "feature_set": fs_name, "n_features": len(feats), "features": ",".join(feats), "n_train_rows": len(dev), "n_test_rows": len(te),
                                   "split": "site-grouped, stratified (test = held-out sites)", **{f"hp_{k}": str(v) for k, v in params.items()}})
                mlflow.log_metrics({f"test_{k}": float(v) for k, v in m.items()})
                with tempfile.TemporaryDirectory() as t:
                    rep = pd.DataFrame(classification_report(y[te], pred, labels=np.arange(K), target_names=le.classes_, output_dict=True, zero_division=0)).T
                    rep.round(4).to_csv(f"{t}/classification_report_test.csv"); mlflow.log_artifact(f"{t}/classification_report_test.csv")
                    json.dump({"features": feats, "classes": list(le.classes_)}, open(f"{t}/model_info.json", "w")); mlflow.log_artifact(f"{t}/model_info.json")
                mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
                results.append(dict(model=name, features=fs_name, **m)); print(f"{name:20s} {fs_name:10s} macroF1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} top3={m['top3']:.4f}")
    os.makedirs("outputs", exist_ok=True); pd.DataFrame(results).round(5).to_csv("outputs/mlflow_run_summary.csv", index=False)


if __name__ == "__main__":
    main()
