# Databricks notebook source
# MAGIC %md
# MAGIC # Fish genus capstone - MLflow tracking on Databricks
# MAGIC Run from a Databricks Git folder linked to this repository. Runs are logged to the built-in MLflow experiment.

# COMMAND ----------

# MAGIC %pip install -q mlflow cloudpickle scikit-learn pandas

# COMMAND ----------

import os, sys, platform, sklearn, mlflow, pyspark
repo = os.path.dirname(os.getcwd())            # notebook lives in <repo>/notebooks
os.chdir(repo); sys.path.insert(0, os.path.join(repo, "src"))
print("Python", platform.python_version(), "| Spark", spark.version, "| MLflow", mlflow.__version__, "| scikit-learn", sklearn.__version__)
print("Databricks runtime:", os.environ.get("DATABRICKS_RUNTIME_VERSION"))

# COMMAND ----------

# Optional: show the Spark side (reads the CSV with Spark to illustrate scaling)
sdf = spark.read.csv(f"file:{repo}/data/cleaned_dataset_expanded.csv", header=True, inferSchema=True)
print(sdf.count(), "rows,", len(sdf.columns), "columns")
display(sdf.groupBy("GENUS").count().orderBy("count", ascending=False).limit(10))

# COMMAND ----------

import runpy, sys
sys.argv = ["train_mlflow.py", "--experiment", f"/Users/{spark.sql('select current_user()').first()[0]}/fish-genus-capstone"]
runpy.run_path("src/train_mlflow.py", run_name="__main__")

# COMMAND ----------

import pandas as pd
display(pd.read_csv("outputs/mlflow_run_summary.csv"))
