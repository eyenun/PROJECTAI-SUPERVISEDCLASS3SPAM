import joblib
import pandas as pd
from utils import split_symptoms


# Load model dan encoder
model = joblib.load(r"D:\PROJEK UAS AI PRAK\models\disease_model.pkl")
le_gender = joblib.load(r"D:\PROJEK UAS AI PRAK\models\le_gender.pkl")
le_disease = joblib.load(r"D:\PROJEK UAS AI PRAK\models\le_disease.pkl")
vectorizer = joblib.load(r"D:\PROJEK UAS AI PRAK\models\vectorizer_symptoms.pkl")

# Fungsi prediksi
def predict_disease(gender, age, symptoms):

    # Encode gender
    gender_encoded = le_gender.transform([gender])[0]

    # Vectorize symptoms
    symptoms_vec = vectorizer.transform([symptoms]).toarray()[0]

    # Gabungkan fitur ke DataFrame satu baris
    data = pd.DataFrame([ [gender_encoded, age] + list(symptoms_vec) ])

    # ==== DEBUG: CEK SHAPE FITUR ====
    print("gender_encoded :", gender_encoded)
    print("age            :", age)
    print("symptoms_vec shape :", symptoms_vec.shape)
    print("data final shape   :", data.shape)
    # =================================



    # Prediksi
    pred = model.predict(data)[0]

    # Decode disease
    return le_disease.inverse_transform([pred])[0]

# Contoh testing manual
hasil = predict_disease("Male", 25, "headache, nausea, dizziness")
print("Model memprediksi penyakit:", hasil)
