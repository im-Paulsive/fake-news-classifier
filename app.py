import streamlit as st
import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Load the trained model and vectorizer
# ---------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------------------
# 2️⃣ Functions to save/load persistent counts
# ---------------------------
def load_counts():
    """Load stored counts from counts.json if it exists."""
    if os.path.exists("counts.json"):
        with open("counts.json", "r") as f:
            return json.load(f)
    else:
        return {"True News": 0, "Fake News": 0}

def save_counts(counts):
    """Save updated counts to counts.json."""
    with open("counts.json", "w") as f:
        json.dump(counts, f)

# Load existing counts
counts = load_counts()

# ---------------------------
# 3️⃣ Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.markdown("Enter a news headline or paragraph below to check whether it's **True** or **Fake**.")

user_input = st.text_area("🗞️ Enter News Text", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Convert input text to TF-IDF
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        prob = np.max(model.predict_proba(input_tfidf))

        if prediction == 1:
            st.success(f"✅ **True News** — Confidence: {prob:.2f}")
            counts["True News"] += 1
        else:
            st.error(f"🚨 **Fake News** — Confidence: {prob:.2f}")
            counts["Fake News"] += 1

        # Save updated counts
        save_counts(counts)

# ---------------------------
# 4️⃣ Show persistent pie chart of total predictions
# ---------------------------
st.subheader("📊 Overall Prediction Summary")

total_predictions = counts["True News"] + counts["Fake News"]

if total_predictions == 0:
    st.info("No predictions yet! Start testing your news to see results here.")
else:
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = ["#4CAF50", "#F44336"]  # green for true, red for fake

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 12},
    )
    ax.axis("equal")
    st.pyplot(fig)

    st.write(f"🟢 **True News:** {counts['True News']}  |  🔴 **Fake News:** {counts['Fake News']}")
