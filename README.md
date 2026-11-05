# Fish Genus Prediction from Stream Data - MLOps (Capstone Milestone 5)

**DATA4402 Data Science Capstone - Austin Novak**

## Project Overview
Can fish genus be predicted from stream location and water conditions? The project builds supervised ML models on public EPA wadeable-stream data (Mid-Atlantic / Appalachian states, 1990-2014), explains them with SHAP, selects extra features with RFE and permutation importance, tunes models with grouped GridSearchCV, and (this milestone) tracks everything with **MLflow**, versions it on **GitHub**, validates it with **GitHub Actions CI/CD**, and runs on **Databricks**.

Honest headline: the task is hard. About nine genera share each site's measurements, so a model that sees only site conditions cannot separate them. On held-out test sites the best model (Random Forest, 13 features) reaches **macro-F1 0.052, accuracy 0.075, Top-3 0.231** (always guessing the most common genus: accuracy 0.125 but macro-F1 0.004). Adding features chosen inside each training fold helped a little; tuning alone helped very little.

## Dataset
`data/cleaned_dataset.csv` - 42,144 fish records, 3,278 stream sites, 50 genera, 7 states (cleaned in Milestones 2-3).
`data/cleaned_dataset_expanded.csv` - same rows with 27 candidate feature columns (original 7 + 20 extras such as `Temp`, `LWSKM2`, `YEAR`, `HUC`).
Source: U.S. EPA Office of Research and Development, *Combined Fish Data* (Data.gov).

## Methodology
- **Split:** by stream site (all rows of a site stay together), stratified by genus. Held-out test sites are used once, at the end.
- **Preprocessing:** IQR capping of chemistry outliers (never deleted), median imputation with missing-flags, scaling, one-hot encoding - fit inside training data only.
- **SHAP (Milestone 4 revision):** location (lat/lon, state, ecoregion) drives 69-82% of each model's importance.
- **Feature selection:** RFE + permutation importance with random-noise probes, run inside each outer training fold. Additions: LWSKM2, Temp, YEAR, HUC (stable), NO3, alk (weak).
- **Tuning:** GridSearchCV, 5-fold StratifiedGroupKFold grouped by `SITE_ID`, scored by macro-F1; nested inside outer folds for a fair original-vs-expanded comparison.
- **MLflow (this milestone):** `src/train_mlflow.py` logs 8 runs (4 models x original-7 / expanded-13 features) with parameters, test metrics, classification-report and model-info artifacts, and the fitted model.

| Model | Features | Test macro-F1 | Test accuracy | Test Top-3 |
|---|---|---|---|---|
| Random Forest | original 7 | 0.0491 | 0.0712 | 0.2177 |
| Random Forest | expanded 13 | **0.0522** | **0.0751** | **0.2313** |
| Gradient Boosting | original 7 / expanded | 0.0443 / 0.0496 | 0.0579 / 0.0649 | 0.1824 / 0.2084 |
| Logistic Regression | original 7 / expanded | 0.0283 / 0.0359 | 0.0440 / 0.0532 | 0.1294 / 0.1631 |
| Neural Network | original 7 / expanded | 0.0317 / 0.0389 | 0.0523 / 0.0667 | 0.1453 / 0.1915 |

Full results, nested-CV tables, figures and SHAP outputs are in `outputs/`.

## Tools & Technologies
Python, pandas, scikit-learn, SHAP, MLflow, Databricks (Git folders + built-in MLflow), GitHub, GitHub Actions, Streamlit and Docker (optional extras).

## Repository Structure
```
data/        cleaned datasets            notebooks/  executed Milestone 4 notebook + Databricks training notebook
src/         pipeline.py, train_mlflow.py, m4_utils.py      models/   saved tuned pipelines
outputs/     results CSVs, figures, MLflow run summary        tests/    pytest checks used by CI
.github/workflows/ci.yml   GitHub Actions        app.py, Dockerfile   optional Streamlit demo
```

## How to Run the Project
1. Clone the repository: `git clone <repo-url> && cd <repo>`
2. Install dependencies: `pip install -r requirements.txt`
3. Train and track with MLflow: `PYTHONPATH=src python src/train_mlflow.py` (about 1 minute; add `--sample 300` for a quick run)
4. View runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db` then open http://localhost:5000
5. Run the checks CI runs: `PYTHONPATH=src pytest -q`
6. Optional demo: `streamlit run app.py` (or `docker build -t fish-demo . && docker run -p 8501:8501 fish-demo`)

On Databricks: add this repo as a Git folder, open `notebooks/databricks_mlflow_training.py`, attach a cluster, Run all.

## CI/CD
`.github/workflows/ci.yml` runs on every push and pull request: installs `requirements.txt`, validates the file structure, runs `pytest` (site-level split has no leakage, pipeline trains, saved model loads and predicts), and smoke-tests the MLflow training script on a sample of sites.

## Results
See the table above and `outputs/results/`. Tuning alone changed macro-F1 by about +0.002 on average; expanded features gave about +0.005 macro-F1 and +1.4 points Top-3 in nested CV. Absolute scores remain low.

## Future Improvements and Scalability
- Add land-use, elevation and stream-order predictors (EPA StreamCat / NHDPlus) and predict a ranked list of likely genera per site.
- Scale: Spark reads and distributes the data; `GridSearchCV` can be parallelised with Spark/Joblib; MLflow Model Registry for versions and stage promotion; Databricks Jobs for scheduled retraining; serve the registered model as a REST endpoint.
- Add DVC for data versioning and a drift check on incoming data.

## Added after feedback
- SHAP beeswarm plots: outputs/shap/
- Region-wise models (feature availability by state, regional vs global, bootstrap): outputs/regional/, code in src/extra/
