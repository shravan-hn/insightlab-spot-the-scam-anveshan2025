# Spot The Scam — Fraudulent Job Posting Detection

## 🧠 Overview

Detect fake job postings and protect users **before** they apply.  
Our solution uses Machine Learning to scan uploaded job listings and flag potential scams — fast, accurate, and fully automated.

---

## 🎯 Problem Statement

Online job platforms are increasingly targeted by scammers. These fake job listings waste applicant time and expose users to **data theft** and **financial loss**.

Our solution automatically detects potentially fraudulent job postings from any CSV input file using a trained machine learning model.

---

## 🛠️ Solution Approach

- Binary classification: **Fraudulent** vs **Genuine**
- Text preprocessing and feature extraction using **TF-IDF**
- Model trained using **Logistic Regression** with class weighting
- Targeted for **high F1-score** (due to class imbalance)

---

## ✨ Key Features

- ✅ Accepts any uploaded CSV (any file name)
- ⚙️ Fully automated preprocessing & prediction
- 📊 Outputs fraud **probabilities per row**
- 📁 Saves results into new CSV file
- 📈 Integrated Streamlit dashboard for insights
- 🔁 Modular pipeline for easy retraining or extension
- 📧 Email alert for high-risk jobs (optional)
- 🔍 Future-ready: supports SHAP explainability

---

## 📂 Folder Structure

```

spot-the-scam-ml/
├── preprocessing.py
├── train\_model.py
├── predict\_csv.py
├── email\_alert.py         # (optional)
├── model/
│   ├── vectorizer.pkl
│   └── fraud\_detector.pkl
├── data/
│   └── (your CSV files go here)
├── app.py                 # Streamlit Dashboard
├── requirements.txt
└── README.md

````

---

## 🧪 Dataset

- Based on Kaggle-style job listings dataset
- Fields include: `title`, `company_profile`, `description`, `requirements`, `benefits`, etc.
- **Imbalanced** dataset with ~5% fraudulent jobs

---

## ⚙️ How to Run Locally

1️⃣ Install dependencies  
```bash
pip install -r requirements.txt
````

2️⃣ Train the model

```bash
python train_model.py
```

3️⃣ Predict on a new CSV file

```bash
python predict_csv.py your_input.csv your_output.csv
```

4️⃣ Launch the dashboard

```bash
streamlit run app.py
```

---

## 📈 Model Performance

* **F1-Score**: \~0.82
* Optimized for class imbalance and real-world generalization

---

## 🔔 Optional: Email Alert Feature

* Automatically emails high-risk job listings after prediction
* Uses Gmail App Password or SMTP setup (configurable)

---

## 🚀 Live App Demo

* YouTube Walkthrough: [📺 Watch here](https://youtu.be/iGOTGmxwQYM)

---

## 👨‍💻 Team

* 🤖  **Ved Sojitra**
* 🌐  **Saravana Ganesh**
* 📊  **Shravan H N**

---

## Thank You!

This system helps job platforms **stop fraud at the source** — protecting users and improving trust.

---

