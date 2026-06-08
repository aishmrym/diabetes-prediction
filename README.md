# 🩺 Diabetes Prediction using Machine Learning

Proyek Akhir Mata Kuliah **Kecerdasan Artifisial Lanjut**

Proyek ini bertujuan untuk membangun model machine learning yang mampu memprediksi kemungkinan seseorang mengalami diabetes berdasarkan data kesehatan pasien menggunakan **Pima Indians Diabetes Dataset**. Proyek mencakup tahapan preprocessing data, pelatihan model klasifikasi, evaluasi performa model, perbandingan beberapa algoritma machine learning, serta implementasi dashboard interaktif menggunakan Streamlit.

---

# 👥 Anggota Kelompok 6

| Nama | Tanggung Jawab |
|--------|--------|
| Sabrina Khairunnisa | Data Cleaning, Preprocessing Data, Dokumentasi Dataset |
| Shafa Rizwana | Exploratory Data Analysis (EDA), Visualisasi Data |
| Anindhita Faiza | Training Model Machine Learning |
| Aisha Maryam | Evaluasi Model, Analisis Metrik, Integrasi dengan Dashboard Streamlit |
| Latifah Puti | Hyperparameter Tuning dan Perbandingan Model |

---

# 📂 Struktur Repository

```text
diabetes-prediction/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── diabetes.csv
│
├── model/
│   ├── diabetes_model.pkl
│   ├── scaler.pkl
│   └── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tuning.py
│   └── predict.py
│
└── dashboard/
    └── app.py
```

---

# 📊 Dataset

Dataset yang digunakan adalah **Pima Indians Diabetes Dataset** yang tersedia melalui UCI Machine Learning Repository dan Kaggle.

### Informasi Dataset

- Jumlah data : 768 observasi
- Jumlah fitur : 8 fitur numerik
- Target : Outcome

Keterangan target:

- 0 = Tidak Diabetes
- 1 = Diabetes

### Fitur yang Digunakan

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

---

# ⚙️ Tahapan Machine Learning

## 1. Data Preprocessing

Tahapan preprocessing dilakukan pada file `preprocessing.py`.

Proses yang dilakukan:

- Membaca dataset diabetes.csv
- Mengidentifikasi nilai 0 yang tidak valid pada beberapa fitur medis
- Mengubah nilai 0 menjadi missing value (NaN)
- Melakukan imputasi missing value menggunakan rata-rata berdasarkan kelas Outcome
- Memisahkan fitur dan target
- Melakukan normalisasi menggunakan StandardScaler

Kolom yang diproses:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

---

## 2. Data Splitting

Dataset dibagi menjadi:

- Training Data : 80%
- Testing Data : 20%

Menggunakan:

```python
train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Penggunaan `stratify=y` bertujuan menjaga proporsi kelas diabetes dan non-diabetes tetap seimbang pada data train maupun test.

---

## 3. Training Model

Pelatihan model dilakukan pada file `train.py`.

Empat algoritma klasifikasi yang digunakan:

### Logistic Regression

Digunakan sebagai baseline model klasifikasi linear.

### K-Nearest Neighbors (KNN)

Menggunakan:

- n_neighbors = 3
- weights = distance
- Manhattan Distance (p = 1)

### Decision Tree

Menggunakan:

- criterion = entropy
- max_depth = 4
- min_samples_split = 4

### Random Forest

Menggunakan:

- n_estimators = 200
- max_depth = 6
- min_samples_split = 4
- max_features = sqrt

Model Random Forest kemudian disimpan dalam bentuk file:

```text
model/diabetes_model.pkl
```

Scaler hasil preprocessing juga disimpan:

```text
model/scaler.pkl
```

---

## 4. Evaluasi Model

Evaluasi dilakukan pada file `evaluate.py`.

Metrik evaluasi yang digunakan:

- Accuracy
- Precision
- Recall
- F1-Score

---

# 📈 Hasil Evaluasi Model

| Model | Accuracy | Precision | Recall | F1-Score |
|---------|---------|---------|---------|---------|
| Logistic Regression | 75.97% | 61.97% | 81.48% | 70.40% |
| KNN | 85.71% | 79.63% | 79.63% | 79.63% |
| Decision Tree | 89.61% | 78.79% | 96.30% | 86.67% |
| Random Forest | 85.06% | 76.27% | 83.33% | 79.65% |

### Model Terbaik

Berdasarkan hasil pengujian, model **Decision Tree** memperoleh akurasi tertinggi sebesar **89.61%**.

Model **Random Forest** dipilih untuk implementasi dashboard karena memberikan performa yang baik, lebih stabil terhadap variasi data, serta mampu menghasilkan probabilitas prediksi yang digunakan dalam visualisasi tingkat risiko diabetes.

---

## 5. Perbandingan Model

File `tuning.py` digunakan untuk membandingkan performa model melalui visualisasi accuracy menggunakan bar chart.

Model yang dibandingkan:

- Logistic Regression
- KNN
- Decision Tree
- Random Forest

Visualisasi dibuat menggunakan:

- Matplotlib
- Seaborn

---

# 🖥️ Dashboard Streamlit

Dashboard dibangun menggunakan Streamlit pada file:

```text
dashboard/app.py
```

Fitur dashboard:

- Input data pasien secara interaktif
- Prediksi diabetes secara real-time
- Menampilkan probabilitas diabetes
- Menampilkan tingkat risiko (Low, Medium, High)
- Menampilkan hasil akhir prediksi:
  - Terindikasi Diabetes
  - Tidak Terindikasi Diabetes
- Menampilkan informasi model dan dataset

---

# 🚀 Cara Menjalankan Proyek

## Clone Repository

```bash
git clone https://github.com/aishmrym/diabetes-prediction.git
cd diabetes-prediction
```

## Install Dependency

```bash
pip install -r requirements.txt
```

## Training Model

```bash
python src/train.py
```

Output:

```text
✅ Preprocessing selesai
✅ Training selesai
✅ Model berhasil disimpan
```

## Evaluasi Model

```bash
python src/evaluate.py
```

Output:

```text
Accuracy
Precision
Recall
F1-Score
```

untuk masing-masing model.

## Menjalankan Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 🛠️ Library yang Digunakan

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit
- Pickle

---

# 🎯 Tujuan Proyek

- Menerapkan konsep machine learning klasifikasi pada kasus kesehatan.
- Membandingkan performa beberapa algoritma klasifikasi.
- Mengevaluasi model menggunakan berbagai metrik evaluasi.
- Mengimplementasikan model machine learning ke dalam dashboard interaktif.
- Memberikan prediksi awal risiko diabetes berdasarkan data pasien.

---

# ⚠️ Disclaimer

Hasil prediksi yang diberikan oleh sistem ini hanya digunakan untuk tujuan edukasi dan pembelajaran machine learning.

Prediksi yang dihasilkan **bukan diagnosis medis resmi** dan tidak dapat menggantikan konsultasi dengan tenaga kesehatan profesional.

---

# 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik dalam Mata Kuliah **Kecerdasan Artifisial Lanjut**.
