import streamlit as st
import joblib

# -------------------------
# Load saved models
# -------------------------
nb_model = joblib.load("naive_bayes.pkl")
hybrid_model = joblib.load("hybrid_dt.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit config
st.set_page_config(page_title="Fake News Detection", page_icon="📰")
st.title("📰 Fake News Detection ")

# User input
user_input = st.text_area("Enter a news article to check:")

# Prediction button
if st.button("Check News"):
    if user_input.strip():
        # Transform input using TF-IDF
        input_features = vectorizer.transform([user_input])

        # Naive Bayes prediction
        pred_nb = nb_model.predict(input_features)[0]
        prob_nb = nb_model.predict_proba(input_features)[0]

        # Hybrid NB→DT prediction
        nb_probs_input = nb_model.predict_proba(input_features)  # probabilities as features
        pred_hybrid = hybrid_model.predict(nb_probs_input)[0]
        prob_hybrid = hybrid_model.predict_proba(nb_probs_input)[0]

        # -------------------------
        # Display results
        # -------------------------
        st.subheader("🧮 Naive Bayes Prediction")
        if pred_nb == 1:
            st.success(f"✅ Real News (Confidence: {prob_nb[1]:.2f})")
        else:
            st.error(f"🚨 Fake News (Confidence: {prob_nb[0]:.2f})")

        st.subheader("🌟 Hybrid NB→DT Prediction")
        if pred_hybrid == 1:
            st.success(f"✅ Real News (Confidence: {prob_hybrid[1]:.2f})")
        else:
            st.error(f"🚨 Fake News (Confidence: {prob_hybrid[0]:.2f})")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
