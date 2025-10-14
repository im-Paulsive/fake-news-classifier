import streamlit as st
import joblib
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Load Models & Vectorizer
# ---------------------------
st.title("📰 Fake News Detection App")
st.caption("Naive Bayes + Hybrid NB→DT Classifier")

try:
    nb_model = joblib.load("naive_bayes.pkl")
    hybrid_model = joblib.load("hybrid_dt.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    st.success("✅ Models and Vectorizer loaded successfully!")
except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e.filename}. Please make sure all .pkl files are in the repo.")
    st.stop()

# ---------------------------
# 2️⃣ Initialize Count File
# ---------------------------
count_file = "count.json"

if not os.path.exists(count_file):
    with open(count_file, "w") as f:
        json.dump({"Real": 0, "Fake": 0}, f)

with open(count_file, "r") as f:
    counts = json.load(f)

# ---------------------------
# 3️⃣ News Input
# ---------------------------
st.subheader("🧾 Enter News Article Text")
news_input = st.text_area("Paste or type the news content below:")

if st.button("🔍 Classify News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text before classifying.")
    else:
        # Transform text
        X_input = vectorizer.transform([news_input])

        # Naive Bayes
        nb_pred = nb_model.predict(X_input)[0]
        nb_prob = nb_model.predict_proba(X_input)[0][1]

        # Hybrid Model (NB→DT)
        nb_test_probs = nb_model.predict_proba(X_input)
        hybrid_pred = hybrid_model.predict(nb_test_probs)[0]
        hybrid_pro_
