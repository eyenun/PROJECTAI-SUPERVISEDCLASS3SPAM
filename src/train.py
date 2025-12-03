import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from utils import split_symptoms
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
df = pd.read_csv(r"D:\PROJEK UAS AI PRAK\data\processed\healthcare_processed.csv")

# Load encoder & vectorizer
le_gender = joblib.load(r"D:\PROJEK UAS AI PRAK\models\le_gender.pkl")
le_disease = joblib.load(r"D:\PROJEK UAS AI PRAK\models\le_disease.pkl")
vectorizer = joblib.load(r"D:\PROJEK UAS AI PRAK\models\vectorizer_symptoms.pkl")

# Pisahkan fitur & label
X = df.drop(columns=["Disease"])
y = df["Disease"]

# Proses kolom Symptoms
symptoms_vectorized = vectorizer.transform(X["Symptoms"])
symptoms_df = pd.DataFrame(symptoms_vectorized.toarray())

# Hapus kolom symptoms lama
X_numeric = X.drop(columns=["Symptoms"]).reset_index(drop=True)

# Gabungkan
X_final = pd.concat([symptoms_df, X_numeric], axis=1)

# FIX WAJIB → Semua column name jadi string
X_final.columns = X_final.columns.astype(str)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, r"D:\PROJEK UAS AI PRAK\models\disease_model.pkl")


print("Training selesai!")
