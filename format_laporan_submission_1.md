# Laporan Proyek Machine Learning - Rakha Apta Pradhana D R

## Domain Proyek

Diabetes mellitus adalah sekelompok penyakit metabolik kronis yang ditandai dengan kadar glukosa darah tinggi (hiperglikemia) yang berkelanjutan. Kondisi ini timbul akibat kelainan pada sekresi insulin, kerja insulin (resistensi insulin), atau keduanya. Menurut International Diabetes Federation (IDF), pada tahun 2021, diperkirakan 537 juta orang dewasa (20-79 tahun) hidup dengan diabetes di seluruh dunia, dan angka ini diproyeksikan meningkat menjadi 643 juta pada tahun 2030 dan 783 juta pada tahun 2045 [1]. Di Indonesia, Riset Kesehatan Dasar (Riskesdas) 2018 menunjukkan prevalensi diabetes melitus pada penduduk usia ≥15 tahun sebesar 2% [2].

**Mengapa dan Bagaimana Masalah Harus Diselesaikan:**
Deteksi dini dan diagnosis diabetes yang akurat sangat krusial karena beberapa alasan utama:
1.  **Pencegahan Komplikasi:** Diabetes yang tidak terdiagnosis atau tidak terkontrol dengan baik dapat menyebabkan komplikasi serius jangka panjang, seperti penyakit kardiovaskular (penyakit jantung koroner, stroke), kerusakan saraf (neuropati diabetik), penyakit ginjal kronis (nefropati diabetik) yang dapat berujung pada gagal ginjal, kerusakan mata (retinopati diabetik) yang dapat menyebabkan kebutaan, dan masalah kaki (ulkus diabetik) yang dapat berujung pada amputasi [3]. Intervensi dini melalui perubahan gaya hidup dan/atau pengobatan dapat secara signifikan mengurangi risiko komplikasi ini.
2.  **Peningkatan Kualitas Hidup:** Manajemen diabetes yang efektif memungkinkan individu untuk menjalani hidup yang lebih sehat dan produktif.
3.  **Pengurangan Beban Sistem Kesehatan:** Biaya perawatan komplikasi diabetes jauh lebih tinggi dibandingkan biaya pencegahan dan manajemen dini. Deteksi dini dapat mengurangi kebutuhan akan perawatan yang mahal dan kompleks di kemudian hari.

Masalah ini dapat diselesaikan dengan mengembangkan sistem pendukung keputusan klinis berbasis machine learning. Model machine learning dapat menganalisis pola kompleks dalam data pasien (atribut medis dan gaya hidup) untuk mengidentifikasi individu yang berisiko tinggi menderita diabetes. Sistem seperti ini dapat berfungsi sebagai alat skrining awal yang efisien, membantu tenaga medis memprioritaskan pasien untuk pemeriksaan lebih lanjut dan intervensi preventif.

**Hasil Riset Terkait atau Referensi:**
Banyak penelitian telah mengeksplorasi penggunaan machine learning untuk prediksi diabetes. Sebagai contoh:
* Maniruzzaman et al. (2018) mengembangkan model prediksi diabetes menggunakan beberapa algoritma machine learning dan menemukan bahwa Random Forest mencapai akurasi yang tinggi [4].
* Sisodia & Sisodia (2018) melakukan studi perbandingan berbagai algoritma klasifikasi untuk prediksi diabetes dan menyoroti pentingnya pemilihan fitur [5].
* Penelitian lain juga menunjukkan potensi algoritma seperti Support Vector Machines (SVM), Logistic Regression, dan Neural Networks dalam domain ini, seringkali dengan fokus pada peningkatan akurasi dan interpretasi model [6].

Pengembangan model yang tidak hanya akurat tetapi juga memiliki recall yang tinggi untuk kasus positif (mendeteksi penderita diabetes) sangat penting, mengingat konsekuensi dari false negative (pasien diabetes yang terlewat) lebih berat.

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
[1] International Diabetes Federation. *IDF Diabetes Atlas, 10th edn.* Brussels, Belgium: 2021. Tersedia: (https://www.diabetesatlas.org)
[2] Kementerian Kesehatan Republik Indonesia. *Laporan Nasional Riskesdas 2018.* Jakarta: Lembaga Penerbit Badan Penelitian dan Pengembangan Kesehatan, 2019.
[3] American Diabetes Association. "Standards of Medical Care in Diabetes—2023." *Diabetes Care*, vol. 46, Supplement 1, 2023.
[4] Maniruzzaman, M., Rahman, M. J., Al-MehediHasan, M., Suri, H. S., Abedin, M. M., El-Baz, A., ... & Suri, J. S. (2018). "Accurate diabetes risk stratification using machine learning: role of missing value and outliers." *Journal of medical systems*, 42(5), 1-15.
[5] Sisodia, D., & Sisodia, D. S. (2018). "Prediction of diabetes using classification algorithms." *Procedia computer science*, 132, 1578-1585.
