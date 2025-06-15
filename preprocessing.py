
import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def combine_columns(df):
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
    df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
    df['combined_text'] = df['combined_text'].apply(clean_text)
    return df
