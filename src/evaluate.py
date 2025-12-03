import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load model
with open(r'D:\PROJEK UAS AI PRAK\models\disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data yang sudah kamu split
with open('models/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('models/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
