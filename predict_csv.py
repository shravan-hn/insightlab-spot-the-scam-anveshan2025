
import pandas as pd
import joblib
from preprocessing import combine_columns

def predict_csv(input_csv, output_csv):
    model = joblib.load("model/fraud_detector.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    data = pd.read_csv(input_csv)
    data = combine_columns(data)
    X_new = vectorizer.transform(data['combined_text'])
    pred_probs = model.predict_proba(X_new)[:, 1]
    pred_class = model.predict(X_new)
    data['Fraud_Probability'] = pred_probs
    data['Fraud_Prediction'] = pred_class
    data.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv}")
