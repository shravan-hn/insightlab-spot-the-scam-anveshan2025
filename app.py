# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import combine_columns

# Streamlit Config
st.set_page_config(page_title="Spot the Scam", layout="wide")
st.title("🚨 Spot the Scam - Fake Job Detector")

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("model/fraud_detector.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Tabs layout
tabs = st.tabs(["🏠 Home", "📊 Analyze Results", "📈 Distribution", "🚩 Top 10 Suspects"])


def get_bar(percent):
    blocks = int(percent // 5)  # 20 blocks = 100%
    return "█" * blocks + "░" * (20 - blocks)

# 🏠 Home Tab
with tabs[0]:
    st.header("Upload Job Listings CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = combine_columns(df)
        X = vectorizer.transform(df['combined_text'])

        df['Fraud_Probability'] = model.predict_proba(X)[:, 1]
        df['Fraud_Prediction'] = model.predict(X)

        st.success("✅ Prediction completed!")
        st.session_state["results"] = df
        st.info("📊 Switch to the 'Analyze Results' or 'Distribution' tabs to explore the predictions.")

# 📊 Analyze Results Tab
with tabs[1]:
    st.header("Prediction Results")
    if "results" in st.session_state:
        df = st.session_state["results"]
        df_display = df.copy()
        df_display['Fraud_Probability (%)'] = (df_display['Fraud_Probability'] * 100).round(2)
        df_display['description'] = df_display['description'].astype(str).str.slice(0, 60) + '...'
        df_display['Confidence Bar'] = df_display['Fraud_Probability (%)'].apply(get_bar)

        st.dataframe(df_display[['title', 'description', 'location', 'Fraud_Probability (%)', 'Confidence Bar']], use_container_width=True)
    else:
        st.info("Please upload a file in the Home tab.")

with tabs[2]:
    st.header("Fraud Probability Distribution")
    if "results" in st.session_state:
        df = st.session_state["results"]

        # Bar chart of binned probabilities
        bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%",
                      "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
        bins = pd.cut(df['Fraud_Probability'], bins=bin_edges, labels=bin_labels, include_lowest=True)
        prob_dist = bins.value_counts().sort_index()

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            prob_dist.plot(kind='bar', color='steelblue', ax=ax1)
            ax1.set_title("Probability Range Distribution")
            ax1.set_xlabel("Range (%)")
            ax1.set_ylabel("Listings")
            fig1.tight_layout()
            st.pyplot(fig1)

        with col2:
            real = (df['Fraud_Prediction'] == 0).sum()
            fake = (df['Fraud_Prediction'] == 1).sum()
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.pie([real, fake], labels=["Real", "Fraud"], autopct="%1.1f%%", colors=["green", "red"])
            ax2.set_title("Real vs Fraud")
            fig2.tight_layout()
            st.pyplot(fig2)
    else:
        st.info("Please upload a file in the Home tab.")

# 🚩 Top 10 Tab
with tabs[3]:
    st.header("Top 10 Most Suspicious Listings")
    if "results" in st.session_state:
        df = st.session_state["results"]
        top10 = df.sort_values(by="Fraud_Probability", ascending=False).head(10).copy()
        top10['description'] = top10['description'].astype(str).str.slice(0, 25) + '...'
        top10['Fraud_Probability (%)'] = (top10['Fraud_Probability'] * 100).round(2)
        st.table(top10[['title', 'description', 'location', 'Fraud_Probability (%)']])
    else:
        st.info("Please upload a file in the Home tab.")
