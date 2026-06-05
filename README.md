# diabetes-prediction
Diabetes prediction web app using Machine Learning (LR, KNN, Decision Tree, Random Forest) — Pima Indians Dataset

# 🩺 Diabetes Prediction — Machine Learning Classification

> Proyek Akhir Mata Kuliah **Kecerdasan Artifisial Lanjut**  
> Dataset: Pima Indians Diabetes | Kelompok 6

---

## 👥 Anggota Kelompok

| Nama | Divisi |
|------|--------|
| Sabrina Khairunnisa | Data Cleaning, Preprocessing, Integrasi Notebook |
| Shafa Rizwana | EDA dan Visualisasi Data |
| Anindhita Faiza | Training Model Machine Learning |
| Aisha Maryam | Evaluasi Model dan Analisis Metrik |
| Latifah Puti | Hyperparameter Tuning & Perbandingan Model |

---

## 📁 Struktur Repository

```
diabetes-prediction/
│
├── data/
│   └── diabetes.csv              # Dataset Pima Indians Diabetes
│
├── notebook/
│   └── diabetes_prediction.ipynb # Notebook Google Colab lengkap
│
├── src/
│   ├── preprocessing.py          # Fungsi cleaning & preprocessing
│   ├── train.py                  # Training & simpan model
│   └── predict.py                # Fungsi prediksi
│
├── model/
│   └── model.pkl                 # Model Random Forest tersimpan
│
├── dashboard/
│   └── app.py                    # Dashboard Streamlit
│
├── requirements.txt              # Daftar library
└── README.md                     # Dokumentasi ini
```

---

## 📊 Dataset

- **Sumber**: Pima Indians Diabetes Database (UCI / Kaggle)
- **Jumlah data**: 768 baris, 9 kolom
- **Fitur**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Outcome (0 = Tidak Diabetes, 1 = Diabetes)

---

## 🤖 Model yang Digunakan

| Model | Keterangan |
|-------|-----------|
| Logistic Regression | Klasifikasi berbasis fungsi sigmoid |
| K-Nearest Neighbors | Klasifikasi berbasis jarak ke tetangga |
| Decision Tree | Pohon keputusan berbasis Gini Impurity |
| Random Forest | Ensemble dari banyak Decision Tree |

---

## 📈 Hasil Evaluasi

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~77% | - | - | - |
| KNN | ~74% | - | - | - |
| Decision Tree | ~72% | - | - | - |
| **Random Forest** | **~79%** | - | - | - |

> *Isi tabel dengan hasil aktual setelah menjalankan notebook*

---

## 🚀 Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/username/diabetes-prediction.git
cd diabetes-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan notebook (Google Colab)
- Upload folder ke Google Drive
- Buka `notebook/diabetes_prediction.ipynb` di Google Colab
- Runtime > Run All

### 4. Jalankan dashboard lokal
```bash
cd dashboard
streamlit run app.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik — Mata Kuliah Kecerdasan Artifisial Lanjut.

