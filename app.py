"""Bonus: tiny Streamlit demo. Pick a stream's conditions and see the 3 most likely fish genera.  Run: streamlit run app.py"""
import json, pickle, sys
import pandas as pd, streamlit as st
sys.path.insert(0, "src")
import m4_utils  # noqa: F401
from pipeline import load

st.title("Stream fish genus predictor (capstone demo)")
info = json.load(open("models/feature_lists.json")); feats = info["expanded"]
model = pickle.load(open("models/tuned_expanded_gradient.pkl", "rb")); le = pickle.load(open("models/label_encoder.pkl", "rb"))
df = load(); row = {}
for f in feats:
    if df[f].dtype == object: row[f] = st.selectbox(f, sorted(df[f].dropna().unique()))
    else: row[f] = st.slider(f, float(df[f].min()), float(df[f].max()), float(df[f].median()))
if st.button("Predict"):
    p = model.predict_proba(pd.DataFrame([row])); top = p[0].argsort()[::-1][:3]
    st.table(pd.DataFrame({"genus": le.classes_[model.classes_[top]], "probability": p[0][top].round(3)}))
st.caption("Research prototype: scores are low (about 23% Top-3 on held-out sites); see README.")
