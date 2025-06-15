# 🚀 Spot The Scam — Fraudulent Job Posting Detection

## Overview

Detect fake job postings and protect users before they apply.

## Problem Statement

Online job platforms are increasingly targeted by scammers. These fake job listings waste applicant time and expose them to financial loss or data theft.
Our solution uses Machine Learning to automatically detect potentially fraudulent job postings from CSV input files.

## Solution Approach

- Binary classification (fraudulent vs genuine)
- Text cleaning, feature extraction, and model training
- Model trained using Logistic Regression and TF-IDF vectorization
- Focus on high F1-score (imbalanced data)

## Key Features

- Accepts any CSV file (any name)
- Fully automated preprocessing pipeline
- Predicts fraud probability for each job
- Generates CSV output with predictions
- Supports further dashboard visualization
- Code fully modular for easy retraining
- Can be extended with email alerts & SHAP explainability

## Dataset

- Kaggle-style job dataset with fields like: title, company_profile, description, requirements, benefits, employment_type, etc.
- Highly imbalanced: ~5% fraudulent

## Technologies Used

- Python 3.12+
- pandas
- scikit-learn
- joblib
- re (regex)
- (optionally: smtplib for email alert)

## Folder Structure

```
spot-the-scam-ml/
├── preprocessing.py
├── train_model.py
├── predict_csv.py
├── email_alert.py  (optional)
├── model/
│   ├── vectorizer.pkl
│   └── fraud_detector.pkl
├── data/
│   └── (your CSV files go here)
├── requirements.txt
└── README.md
```

## How to Run

1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

2️⃣ Train the model

```bash
python train_model.py
```

3️⃣ Predict on new CSV file

```bash
python predict_csv.py your_file.csv your_output.csv
```

## Current F1 Score

F1-score: ~0.82  
Model is designed for maximum generalization on unseen data.

## Optional: Email Alert Feature

If enabled, after prediction it can email high-risk job listings automatically.

## Note

- Email alert uses Google App Password or SMTP provider
- Fully GDPR-safe as no user data is stored

## Team

- ML: Ved Sojitra
- Frontend: Saravana Ganesh
- Dashboard: Shravan H N

## Submission Links

- Video Demo:

## Thank you!
