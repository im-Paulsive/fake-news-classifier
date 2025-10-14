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
        hybrid_prob = hybrid_model.predict_proba(nb_test_probs)[0][1]

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        if nb_pred == 1:
            st.success(f"🧮 Naive Bayes: ✅ Real News (Confidence: {nb_prob:.2f})")
        else:
            st.error(f"🧮 Naive Bayes: 🚨 Fake News (Confidence: {nb_prob:.2f})")

        if hybrid_pred == 1:
            st.success(f"🌟 Hybrid NB→DT: ✅ Real News (Confidence: {hybrid_prob:.2f})")
            counts["Real"] += 1
        else:
            st.error(f"🌟 Hybrid NB→DT: 🚨 Fake News (Confidence: {hybrid_prob:.2f})")
            counts["Fake"] += 1

        with open(count_file, "w") as f:
            json.dump(counts, f, indent=4)

# ---------------------------
# 4️⃣ Pie Chart for Predictions
# ---------------------------
st.markdown("---")
st.subheader("📈 Prediction Summary")

labels = list(counts.keys())
sizes = list(counts.values())
total = sum(sizes)

if total == 0:
    st.info("No predictions made yet — classify some news to see stats!")
else:
    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4CAF50", "#F44336"],
    )
    ax.axis("equal")
    st.pyplot(fig)

# ---------------------------
# 5️⃣ Footer
# ---------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Naive Bayes + Hybrid Decision Tree")
