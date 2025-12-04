# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib
import os

# -----------------------------
# Load dataset
# -----------------------------
data_path = r"D:\PROJEK UAS AI PRAK\data\raw\Healthcare.csv"
df = pd.read_csv(data_path)

# -----------------------------
# Encode Gender
# -----------------------------
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])  # Male=1, Female=0, Other=2
joblib.dump(le_gender, "../models/gender.pkl")  # simpan encoder Gender

# -----------------------------
# Encode target Disease
# -----------------------------
le_disease = LabelEncoder()
df['Disease_encoded'] = le_disease.fit_transform(df['Disease'])
joblib.dump(le_disease, "../models/disease.pkl")  # simpan encoder Disease

# -----------------------------
# Process Symptoms (Bag-of-Words)
# -----------------------------
df['Symptoms_clean'] = df['Symptoms'].str.replace(', ', ' ', regex=False)


vectorizer = CountVectorizer()
X_symptoms = vectorizer.fit_transform(df['Symptoms_clean'])
joblib.dump(vectorizer, "../models/vectorizer_symptom.pkl")



# -----------------------------
# Gabungkan semua fitur
# -----------------------------
X_numeric = df[['Age', 'Gender', 'Symptom_Count']].values
X = hstack([X_numeric, X_symptoms])  # gabungkan numerik + text features
y = df['Disease_encoded'].values

# -----------------------------
# Split train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Opsional: simpan dataset preprocessing
# -----------------------------
processed_folder = "../data/processed/"
os.makedirs(processed_folder, exist_ok=True)

# Simpan versi CSV (hanya kolom asli + encoded target)
df.to_csv(os.path.join(processed_folder, "healthcare_processed.csv"), index=False)

print("Preprocessing selesai!")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print(df.head())
print(df.columns)

