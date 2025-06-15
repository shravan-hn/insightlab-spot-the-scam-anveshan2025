
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from preprocessing import combine_columns

os.makedirs("model", exist_ok=True)

df = pd.read_csv("NqndMEyZakuimmFI copy.csv")
df = combine_columns(df)

if 'fraudulent' not in df.columns:
    raise ValueError("Training data must contain 'fraudulent' column.")

X = df['combined_text']
y = df['fraudulent']

vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_vect, y)

model = LogisticRegression(max_iter=500)
model.fit(X_res, y_res)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

joblib.dump(model, 'model/fraud_detector.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("✅ Model training complete. Models saved in model/ folder.")
