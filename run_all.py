import os
import subprocess

print("\n--- Step 1: Training model ---")
subprocess.run(["python", "train_model.py"])

from predict_csv import predict_csv

print("\n--- Step 2: Predicting on test.csv ---")
predict_csv("NqndMEyZakuimmFI copy.csv", "data/test_predictions.csv")

frontend_path = "data/frontend.csv"
if os.path.exists(frontend_path):
    print("\n--- Step 3: Predicting on frontend.csv ---")
    predict_csv(frontend_path, "data/frontend_predictions.csv")
else:
    print("\n(No frontend.csv found - skipping frontend prediction)")