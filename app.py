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
st.caption("Using Naive Bayes and Hybrid NB→DT Models")

try:
    nb_model = joblib.load("naive_bayes.pkl")
    hybrid_model = joblib.load("hybrid_dt.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    st.success("✅ Models and Vectorizer loaded successfully!")
except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e.filename}. Please make sure all .pkl files are in the repo.")
    st.stop()

# ---------------------------
# 2️⃣ Initialize Count Storage
# ---------------------------
count_file = "count.json"

# If file doesn't exist, create it
if not os.path.exists(count_file):
    with open(count_file, "w") as f:
        json.dump({"Real": 0, "Fake": 0}, f)

# Load the counts
with open(count_file, "r") as f:
    counts = json.load(f)

# ---------------------------
# 3️⃣ News Input Section
# ---------------------------
st.subheader("🧾 Enter News Article Text")
news_input = st.text_area("Paste or type the news content below:")

if st.button("🔍 Classify News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text before classifying.")
    else:
        # Transform input text
        X_input = vectorizer.transform([news_input])

        # Predictions
        nb_pred = nb_model.predict(X_input)[0]
        nb_prob = nb_model.predict_proba(X_input)[0][1]

        nb_test_probs = nb_model.predict_proba(X_input)
        hybrid_pred = hybrid_model.predict(nb_test_probs)[0]
        hybrid_prob = hybrid_model.predict_proba(nb_test_probs)[0][1]

        # Display Results
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        # Naive Bayes Result
        if nb_pred == 1:
            st.success(f"🧮 Naive Bayes Prediction\n✅ Real News (Confidence: {nb_prob:.2f})")
        else:
            st.error(f"🧮 Naive Bayes Prediction\n🚨 Fake News (Confidence: {nb_prob:.2f})")

        # Hybrid Result
        if hybrid_pred == 1:
            st.success(f"🌟 Hybrid NB→DT Prediction\n✅ Real News (Confidence: {hybrid_prob:.2f})")
            counts["Real"] += 1
        else:
            st.error(f"🌟 Hybrid NB→DT Prediction\n🚨 Fake News (Confidence: {hybrid_prob:.2f})")
            counts["Fake"] += 1

        # Update counts file
        with open(count_file, "w") as f:
            json.dump(counts, f, indent=4)

# ---------------------------
# 4️⃣ Show Summary Pie Chart
# ---------------------------
st.markdown("---")
st.subheader("📈 Prediction Summary")

labels = list(counts.keys())
sizes = list(counts.values())

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#4CAF50", "#F44336"])
ax.axis("equal")
st.pyplot(fig)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Naive Bayes + Hybrid Decision Tree")
