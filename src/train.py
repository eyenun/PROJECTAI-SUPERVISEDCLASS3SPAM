# src/train.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

# ---------------------------
# Load data hasil preprocessing
# ---------------------------
df = pd.read_csv(r"D:\PROJEK UAS AI PRAK\data\processed\healthcare_processed.csv")
le_disease = joblib.load(r"D:\PROJEK UAS AI PRAK\models\disease.pkl")

# ---------------------------
# Vectorizer (TF-IDF)
# ---------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_text = vectorizer.fit_transform(df['Symptoms_clean'])

# Simpan vectorizer
joblib.dump(vectorizer, r"D:\PROJEK UAS AI PRAK\models\symptomvectorizer_rf.pkl")

# ---------------------------
# Fitur numerik
# ---------------------------
X_numeric = df[["Age", "Gender", "Symptom_Count"]].values
X_numeric_sparse = csr_matrix(X_numeric)

# Gabungkan numeric + teks
X = hstack([X_numeric_sparse, X_text])
y = df['Disease_encoded'].values

# ---------------------------
# Split train/test
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Training RandomForest
# ---------------------------
model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model.fit(X_train, y_train)  # ❗ Pakai X_train gabungan

# ---------------------------
# Evaluasi
# ---------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------
# Simpan model
# ---------------------------
joblib.dump(model, r"D:\PROJEK UAS AI PRAK\models\diseases_model_rf.pkl")
print("Training selesai! Model siap untuk prediksi manual.")
print("X_train shape:", X_train.shape)
