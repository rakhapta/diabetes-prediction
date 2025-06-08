# Laporan Proyek Machine Learning - Rakha Apta Pradhana D R

## Domain Proyek

Diabetes mellitus adalah sekelompok penyakit metabolik yang ditandai dengan kadar glukosa darah tinggi (hiperglikemia) yang diakibatkan oleh kelainan sekresi insulin, kerja insulin, atau keduanya. Deteksi dini dan diagnosis yang akurat sangat krusial untuk pengelolaan penyakit yang efektif, mencegah komplikasi akut dan kronis, serta meningkatkan kualitas hidup pasien. Komplikasi diabetes dapat mencakup penyakit kardiovaskular, kerusakan saraf (neuropati), kerusakan ginjal (nefropati), kerusakan mata (retinopati), dan masalah kaki. Mengingat dampak signifikan diabetes terhadap kesehatan individu dan sistem layanan kesehatan, pengembangan alat bantu diagnostik yang efisien dan andal menggunakan teknik machine learning menjadi sangat penting. Proyek ini bertujuan untuk membangun model prediktif yang dapat membantu mengidentifikasi individu yang berisiko tinggi menderita diabetes berdasarkan serangkaian atribut medis dan gaya hidup. Fokus utama adalah pada kemampuan model untuk meminimalkan kasus positif yang terlewat (false negative), karena kegagalan mendeteksi diabetes pada tahap awal dapat menunda intervensi yang diperlukan.

## Business Understanding

### Problem Statement
- Bagaimana mengembangkan model machine learning yang dapat memprediksi risiko diabetes pada seseorang berdasarkan faktor-faktor kesehatan tertentu?
- Bagaimana membandingkan performa antara model K-Nearest Neighbors (KNN) dan Random Forest dalam prediksi diabetes?

### Goals
- Mengembangkan model prediktif yang dapat mengidentifikasi risiko diabetes dengan akurasi yang baik
- Membandingkan dan mengevaluasi performa model KNN dan Random Forest untuk menentukan model yang lebih sesuai untuk kasus ini

### Solution Approach
- Menggunakan dua algoritma machine learning: K-Nearest Neighbors (KNN) dan Random Forest
- Melakukan preprocessing data termasuk standardisasi fitur
- Mengevaluasi model menggunakan berbagai metrik seperti confusion matrix dan classification report

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Pima Indians Diabetes Database, yang aslinya berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Tujuan utama dari dataset ini adalah untuk memprediksi secara diagnostik apakah seorang pasien menderita diabetes atau tidak, berdasarkan pengukuran diagnostik tertentu. Terdapat batasan spesifik dalam pemilihan data ini dari database yang lebih besar, yaitu semua pasien adalah wanita berusia minimal 21 tahun dari keturunan Indian Pima(Pima Indian Heritage).

Sumber Data: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Variabel-variabel pada dataset:
1. Pregnancies: Jumlah kehamilan
2. Glucose: Kadar glukosa plasma
3. BloodPressure: Tekanan darah diastolik (mm Hg)
4. SkinThickness: Ketebalan lipatan kulit trisep (mm)
5. Insulin: Insulin serum 2-Jam (mu U/ml)
6. BMI: Indeks massa tubuh
7. DiabetesPedigreeFunction: Riwayat diabetes dalam keluarga
8. Age: Usia dalam tahun
9. Outcome: Variabel target (1: diabetes, 0: tidak diabetes)

### Analisis Data:
- Dataset terdiri dari 768 sampel dengan 8 fitur dan 1 variabel target
- Tidak ada missing values dalam dataset
- Terdapat beberapa nilai 0 yang tidak realistis untuk beberapa fitur seperti Glucose dan BloodPressure
- Data tidak seimbang dengan lebih banyak kasus non-diabetes

## Data Preparation
1. Data Splitting:
   - Data dibagi menjadi training set (80%) dan testing set (20%)
   - Menggunakan stratified split untuk mempertahankan proporsi kelas

2. Standardization:
   - Menggunakan StandardScaler untuk menormalkan fitur
   - Fitur diubah menjadi distribusi dengan mean=0 dan variance=1
   - Scaling diterapkan pada training dan test set secara terpisah

## Modeling
Dua model machine learning digunakan dalam proyek ini:

1. K-Nearest Neighbors (KNN):
   - Menggunakan k=3 nearest neighbors
   - Kelebihan: Sederhana, mudah diimplementasikan
   - Kekurangan: Sensitif terhadap skala fitur, komputasi berat untuk dataset besar

2. Random Forest:
   - Menggunakan 100 trees dengan max_depth=15
   - Kelebihan: Robust terhadap overfitting, dapat menangani fitur non-linear
   - Kekurangan: Memerlukan lebih banyak memori, kurang interpretable

## Evaluation
Evaluasi model menggunakan beberapa metrik:
- Confusion Matrix
- Classification Report (Accuracy, Precision, Recall, F1-Score)

Kedua model menunjukkan performa yang baik, dengan Random Forest sedikit lebih unggul dalam hal akurasi dan recall untuk kelas positif (diabetes).

## Conclusion
- Random Forest menunjukkan performa yang lebih baik dalam prediksi diabetes
- Standardisasi fitur sangat penting untuk performa model KNN
- Model dapat digunakan sebagai alat screening awal untuk risiko diabetes, tetapi tidak menggantikan diagnosis medis profesional

## Referensi
- Dataset: Pima Indians Diabetes Database
- Scikit-learn documentation
- Research papers tentang prediksi diabetes menggunakan machine learning 
