# Proyek UAS Machine Learning

## Deskripsi
Analisis dataset <nama dataset> dari Kaggle menggunakan <algoritma>. 
Model ini dapat menerima input manual dan memberikan prediksi otomatis.

## Struktur Folder
<project-uas-ml/
│
├── data/
│   ├── raw/                 # dataset asli dari Kaggle (tanpa disentuh)
│   ├── processed/           # dataset setelah preprocessing (cleaning, encoding, scaling, dll.)
│   └── external/            # kalau ada file tambahan (lookup, mapping, stopwords, dsb.)
│
├── notebooks/
│   ├── 01_exploration.ipynb # EDA: melihat karakteristik data
│   ├── 02_preprocessing.ipynb
│   └── 03_modelling.ipynb   # training model, tuning, evaluasi
│
├── src/
│   ├── preprocessing.py     # fungsi preprocessing (cleaning teks, encoding, dsb.)
│   ├── train.py             # script untuk train model
│   ├── evaluate.py          # fungsi evaluasi (precision, recall, f1)
│   └── predict.py           # function / script input data sendiri → hasil prediksi
│
├── models/
│   ├── model.pkl            # hasil training
│   └── vectorizer.pkl       # kalau pakai text model (TF-IDF dsb.)
│
├── reports/
│   ├── figures/             # grafik confusion matrix, grafik training, dll.
│   └── laporan_uas.docx     # laporan akhir
│
├── requirements.txt         # library yang dibutuhkan
└── README.md                # penjelasan singkat proyek>

## Cara Menjalankan
1. pip install -r requirements.txt
2. Jalankan notebook di folder `notebooks/`
3. Training model: python src/train.py
4. Prediksi manual: python src/predict.py
