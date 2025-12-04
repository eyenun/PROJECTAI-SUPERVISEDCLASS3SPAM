import joblib
from scipy.sparse import csr_matrix, hstack

# ========================
# Load model & vectorizer yang benar
# ========================
model = joblib.load(r"D:\PROJEK UAS AI PRAK\models\diseases_model_rf.pkl")
vectorizer = joblib.load(r"D:\PROJEK UAS AI PRAK\models\symptomvectorizer_rf.pkl")
le_disease = joblib.load(r"D:\PROJEK UAS AI PRAK\models\disease.pkl")

# ========================
# Fungsi input data pasien
# ========================
def input_data():
    print("\nMasukkan data pasien:")
    age = int(input("Umur: "))
    gender = input("Gender (Male/Female/Other): ").strip().lower()
    if gender == "male":
        gender_val = 1
    elif gender == "female":
        gender_val = 0
    else:
        gender_val = 2
    symptoms = input("Gejala (pisahkan koma): ")
    symptoms_clean = symptoms.replace(",", " ").strip()
    symptom_count = len(symptoms.split(","))  # otomatis hitung jumlah gejala
    return age, gender_val, symptom_count, symptoms_clean

# ========================
# Loop prediksi manual
# ========================
while True:
    age, gender_val, symptom_count, symptoms_clean = input_data()
    
    # Fitur numerik
    X_numeric = csr_matrix([[age, gender_val, symptom_count]])
    
    # Fitur teks (TF-IDF)
    X_text = vectorizer.transform([symptoms_clean])
    
    # Gabungkan numerik + teks → total fitur = 652
    X_input = hstack([X_numeric, X_text])
    
    # Prediksi
    pred_encoded = model.predict(X_input)[0]
    pred_label = le_disease.inverse_transform([pred_encoded])[0]
    
    print(f"\nPrediksi penyakit: {pred_label}\n")
    
    cont = input("Coba data lain? (y/n): ")
    if cont.lower() != "y":
        print("Selesai. Terima kasih!")
        break
