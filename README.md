# 🩺 Diabetes Prediction using Machine Learning

Proyek Akhir Mata Kuliah **Kecerdasan Artifisial Lanjut**

Proyek ini bertujuan untuk membangun model machine learning untuk memprediksi kemungkinan seseorang menderita diabetes berdasarkan data kesehatan pada **Pima Indians Diabetes Dataset**. Proyek mencakup seluruh tahapan machine learning mulai dari preprocessing data, eksplorasi data, pelatihan model, evaluasi performa, hingga implementasi dashboard prediksi menggunakan Streamlit.

---

## 👥 Anggota Kelompok 6

| Nama                | Tanggung Jawab                                  |
| ------------------- | ----------------------------------------------- |
| Sabrina Khairunnisa | Data Cleaning, Preprocessing                    |
| Shafa Rizwana       | Exploratory Data Analysis (EDA) dan Visualisasi |
| Anindhita Faiza     | Training Model Machine Learning                 |
| Aisha Maryam        | Evaluasi Model dan Analisis Metrik              |
| Latifah Puti        | Hyperparameter Tuning dan Perbandingan Model    |

---

## 📂 Struktur Repository

```text
diabetes-prediction/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── diabetes.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tuning.py
│   └── predict.py
│
├── model/
│   └── README.md
│
└── dashboard/
    └── app.py
```

---

## 📊 Dataset

Dataset yang digunakan adalah **Pima Indians Diabetes Dataset** yang tersedia melalui UCI Machine Learning Repository dan Kaggle.

### Informasi Dataset

* Jumlah data: 768 observasi
* Jumlah fitur: 8
* Target: Outcome

  * 0 = Tidak Diabetes
  * 1 = Diabetes

### Fitur

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

---

## ⚙️ Tahapan Machine Learning

### 1. Data Preprocessing

* Mengubah nilai 0 menjadi NaN pada fitur tertentu
* Menangani missing value menggunakan class-based mean imputation
* Normalisasi data menggunakan StandardScaler

### 2. Exploratory Data Analysis

* Distribusi kelas diabetes
* Boxplot untuk mendeteksi outlier
* Correlation heatmap
* Analisis statistik deskriptif

### 3. Training Model

Model yang digunakan:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest

### 4. Evaluasi Model

Metode evaluasi:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix
* Cross Validation

### 5. Hyperparameter Tuning

Dilakukan pada model Random Forest untuk meningkatkan performa model.

---

## 📈 Hasil Evaluasi

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 75.97%   |
| KNN                 | 85.71%   |
| Decision Tree       | 89.61%   |
| Random Forest       | 86.36%   |

### Cross Validation

Mean Cross Validation Score:

```text
0.8763
```

Model dengan performa terbaik berdasarkan pengujian adalah **Decision Tree** dengan akurasi sebesar **89.61%**, sedangkan Random Forest memberikan performa yang lebih stabil berdasarkan hasil cross validation.

---

## 🚀 Cara Menjalankan Proyek

### Clone Repository

```bash
git clone https://github.com/aishmrym/diabetes-prediction.git
cd diabetes-prediction
```

### Install Dependency

```bash
pip install -r requirements.txt
```

### Jalankan Training

```bash
python src/train.py
```

### Jalankan Evaluasi

```bash
python src/evaluate.py
```

### Jalankan Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🛠️ Library yang Digunakan

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Streamlit

---

## 🎯 Tujuan Proyek

* Menerapkan konsep machine learning klasifikasi.
* Membandingkan performa beberapa algoritma klasifikasi.
* Mengevaluasi model menggunakan berbagai metrik evaluasi.
* Mengembangkan dashboard sederhana untuk prediksi diabetes.

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik pada Mata Kuliah **Kecerdasan Artifisial Lanjut**.
