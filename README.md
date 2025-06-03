# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang berdiri sejak tahun 2000, dikenal karena telah mencetak lulusan berkualitas dengan reputasi yang baik. Namun, institusi ini menghadapi tantangan tingginya angka siswa yang tidak menyelesaikan pendidikan (dropout). Untuk menjaga reputasi dan mendukung kesuksesan akademik siswa, Jaya Jaya Institut perlu mendeteksi siswa yang berpotensi dropout sejak dini agar dapat memberikan bimbingan khusus. Proyek ini bertujuan untuk membangun model prediktif menggunakan machine learning untuk mengidentifikasi siswa berisiko dropout berdasarkan data akademik, demografi, dan faktor sosial-ekonomi.

## Permasalahan Bisnis

1. **Tingginya Tingkat Dropout**: Banyak siswa tidak menyelesaikan pendidikan mereka, yang berdampak negatif pada reputasi institusi dan tingkat kelulusan.
2. **Keterlambatan Identifikasi Risiko**: Kurangnya sistem untuk mendeteksi siswa berpotensi dropout secara dini, sehingga intervensi bimbingan tidak dapat dilakukan tepat waktu.
3. **Optimalisasi Sumber Daya**: Perlu alokasi sumber daya yang efisien untuk memberikan bimbingan khusus hanya kepada siswa yang benar-benar berisiko.

## Cakupan Proyek

- **Analisis Data**: Melakukan pembersihan data, penanganan outlier menggunakan metode IQR, dan pembuatan fitur baru (seperti `Interaction_Grade_Failed_Courses`, `Age_Squared`, dll.) untuk meningkatkan performa model.
- **Pemodelan Machine Learning**: Mengembangkan dan membandingkan dua model, yaitu Decision Tree dan Random Forest, dengan optimasi parameter menggunakan GridSearchCV.
- **Evaluasi Model**: Mengevaluasi performa model menggunakan metrik akurasi, presisi, recall, F1-score, dan matriks konfusi untuk memilih model terbaik.
- **Implementasi**: Mengekspor model terbaik ke file `joblib` dan data yang telah dibersihkan ke file CSV untuk digunakan dalam dashboard Streamlit.
- **Visualisasi**: Membuat dashboard interaktif menggunakan Looker Studio untuk memvisualisasikan data dan prediksi dropout.
- **Prototipe Sistem**: Mengembangkan aplikasi Streamlit untuk memprediksi risiko dropout secara real-time berdasarkan input data siswa.

## Persiapan

**Sumber Data**:\
Dataset berasal dari GitHub Dicoding Academy. Dataset ini mencakup informasi siswa seperti status perkawinan, mode pendaftaran, nilai masuk, kualifikasi orang tua, performa akademik semester pertama dan kedua, serta status akhir (Dropout, Enrolled, Graduate), link sumber data : [https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv]

## Data Understanding
## Data Loading

Dataset ini berasal dari institusi pendidikan tinggi dan dikumpulkan dari beberapa basis data yang terpisah, berfokus pada mahasiswa yang terdaftar di berbagai program sarjana seperti agronomi, desain, pendidikan, keperawatan, jurnalisme, manajemen, pelayanan sosial, dan teknologi. Dataset mencakup informasi yang diketahui pada saat pendaftaran mahasiswa (jalur akademik, demografi, dan faktor sosial-ekonomi) serta performa akademik mahasiswa pada akhir semester pertama dan kedua. Data ini digunakan untuk membangun model klasifikasi guna memprediksi dropout dan keberhasilan akademik mahasiswa.

Berikut adalah deskripsi kolom dalam dataset:

| Nama Kolom | Deskripsi |
| --- | --- |
|Marital status | The marital status of the student. (Categorical) 1 – single 2 – married 3 – widower 4 – divorced 5 – facto union 6 – legally separated |
| Application mode | The method of application used by the student. (Categorical) 1 - 1st phase - general contingent 2 - Ordinance No. 612/93 5 - 1st phase - special contingent (Azores Island) 7 - Holders of other higher courses 10 - Ordinance No. 854-B/99 15 - International student (bachelor) 16 - 1st phase - special contingent (Madeira Island) 17 - 2nd phase - general contingent 18 - 3rd phase - general contingent 26 - Ordinance No. 533-A/99, item b2) (Different Plan) 27 - Ordinance No. 533-A/99, item b3 (Other Institution) 39 - Over 23 years old 42 - Transfer 43 - Change of course 44 - Technological specialization diploma holders 51 - Change of institution/course 53 - Short cycle diploma holders 57 - Change of institution/course (International)|
|Application order | The order in which the student applied. (Numerical) Application order (between 0 - first choice; and 9 last choice) |
|Course | The course taken by the student. (Categorical) 33 - Biofuel Production Technologies 171 - Animation and Multimedia Design 8014 - Social Service (evening attendance) 9003 - Agronomy 9070 - Communication Design 9085 - Veterinary Nursing 9119 - Informatics Engineering 9130 - Equinculture 9147 - Management 9238 - Social Service 9254 - Tourism 9500 - Nursing 9556 - Oral Hygiene 9670 - Advertising and Marketing Management 9773 - Journalism and Communication 9853 - Basic Education 9991 - Management (evening attendance)|
|Daytime/evening attendance | Whether the student attends classes during the day or in the evening. (Categorical) 1 – daytime 0 - evening |
|Previous qualification| The qualification obtained by the student before enrolling in higher education. (Categorical) 1 - Secondary education 2 - Higher education - bachelor's degree 3 - Higher education - degree 4 - Higher education - master's 5 - Higher education - doctorate 6 - Frequency of higher education 9 - 12th year of schooling - not completed 10 - 11th year of schooling - not completed 12 - Other - 11th year of schooling 14 - 10th year of schooling 15 - 10th year of schooling - not completed 19 - Basic education 3rd cycle (9th/10th/11th year) or equiv. 38 - Basic education 2nd cycle (6th/7th/8th year) or equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 42 - Professional higher technical course 43 - Higher education - master (2nd cycle) |
|Previous qualification (grade) | Grade of previous qualification (between 0 and 200) |
| Nacionality | The nationality of the student. (Categorical) 1 - Portuguese; 2 - German; 6 - Spanish; 11 - Italian; 13 - Dutch; 14 - English; 17 - Lithuanian; 21 - Angolan; 22 - Cape Verdean; 24 - Guinean; 25 - Mozambican; 26 - Santomean; 32 - Turkish; 41 - Brazilian; 62 - Romanian; 100 - Moldova (Republic of); 101 - Mexican; 103 - Ukrainian; 105 - Russian; 108 - Cuban; 109 - Colombian|
|Mother's qualification | The qualification of the student's mother. (Categorical) 1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 22 - Technical-professional course 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle)|
|Father's qualification | The qualification of the student's father. (Categorical) 1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 13 - 2nd year complementary high school course 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 20 - Complementary High School Course 22 - Technical-professional course 25 - Complementary High School Course - not concluded 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 31 - General Course of Administration and Commerce 33 - Supplementary Accounting and Administration 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle) |
| Mother's occupation | The occupation of the student's mother. (Categorical) 0 - Student 1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers 2 - Specialists in Intellectual and Scientific Activities 3 - Intermediate Level Technicians and Professions 4 - Administrative staff 5 - Personal Services, Security and Safety Workers and Sellers 6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry 7 - Skilled Workers in Industry, Construction and Craftsmen 8 - Installation and Machine Operators and Assembly Workers 9 - Unskilled Workers 10 - Armed Forces Professions 90 - Other Situation 99 - (blank) 122 - Health professionals 123 - teachers 125 - Specialists in information and communication technologies (ICT) 131 - Intermediate level science and engineering technicians and professions 132 - Technicians and professionals, of intermediate level of health 134 - Intermediate level technicians from legal, social, sports, cultural and similar services 141 - Office workers, secretaries in general and data processing operators 143 - Data, accounting, statistical, financial services and registry-related operators 144 - Other administrative support staff 151 - personal service workers 152 - sellers 153 - Personal care workers and the like 171 - Skilled construction workers and the like, except electricians 173 - Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like 175 - Workers in food processing, woodworking, clothing and other industries and crafts 191 - cleaning workers 192 - Unskilled workers in agriculture, animal production, fisheries and forestry 193 - Unskilled workers in extractive industry, construction, manufacturing and transport 194 - Meal preparation assistants |
| Father's occupation | The occupation of the student's father. (Categorical) 0 - Student 1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers 2 - Specialists in Intellectual and Scientific Activities 3 - Intermediate Level Technicians and Professions 4 - Administrative staff 5 - Personal Services, Security and Safety Workers and Sellers 6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry 7 - Skilled Workers in Industry, Construction and Craftsmen 8 - Installation and Machine Operators and Assembly Workers 9 - Unskilled Workers 10 - Armed Forces Professions 90 - Other Situation 99 - (blank) 101 - Armed Forces Officers 102 - Armed Forces Sergeants 103 - Other Armed Forces personnel 112 - Directors of administrative and commercial services 114 - Hotel, catering, trade and other services directors 121 - Specialists in the physical sciences, mathematics, engineering and related techniques 122 - Health professionals 123 - teachers 124 - Specialists in finance, accounting, administrative organization, public and commercial relations 131 - Intermediate level science and engineering technicians and professions 132 - Technicians and professionals, of intermediate level of health 134 - Intermediate level technicians from legal, social, sports, cultural and similar services 135 - Information and communication technology technicians 141 - Office workers, secretaries in general and data processing operators 143 - Data, accounting, statistical, financial services and registry-related operators 144 - Other administrative support staff 151 - personal service workers 152 - sellers 153 - Personal care workers and the like 154 - Protection and security services personnel 161 - Market-oriented farmers and skilled agricultural and animal production workers 163 - Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence 171 - Skilled construction workers and the like, except electricians 172 - Skilled workers in metallurgy, metalworking and similar 174 - Skilled workers in electricity and electronics 175 - Workers in food processing, woodworking, clothing and other industries and crafts 181 - Fixed plant and machine operators 182 - assembly workers 183 - Vehicle drivers and mobile equipment operators 192 - Unskilled workers in agriculture, animal production, fisheries and forestry 193 - Unskilled workers in extractive industry, construction, manufacturing and transport 194 - Meal preparation assistants 195 - Street vendors (except food) and street service providers |
| Admission grade | Admission grade (between 0 and 200) |
| Displaced | Whether the student is a displaced person. (Categorical) 	1 – yes 0 – no |
| Educational special needs | Whether the student has any special educational needs. (Categorical) 1 – yes 0 – no |
|Debtor | Whether the student is a debtor. (Categorical) 1 – yes 0 – no|
|Tuition fees up to date | Whether the student's tuition fees are up to date. (Categorical) 1 – yes 0 – no|
|Gender | The gender of the student. (Categorical) 1 – male 0 – female |
|Scholarship holder | Whether the student is a scholarship holder. (Categorical) 1 – yes 0 – no |
|Age at enrollment | The age of the student at the time of enrollment. (Numerical)|
|International | Whether the student is an international student. (Categorical) 1 – yes 0 – no|
|Curricular units 1st sem (credited) | The number of curricular units credited by the student in the first semester. (Numerical) |
| Curricular units 1st sem (enrolled) | The number of curricular units enrolled by the student in the first semester. (Numerical) |
| Curricular units 1st sem (evaluations) | The number of curricular units evaluated by the student in the first semester. (Numerical) |
| Curricular units 1st sem (approved) | The number of curricular units approved by the student in the first semester. (Numerical) |

**Acknowledgements**  
Realinho, Valentim, Vieira Martins, Mónica, Machado, Jorge, dan Baptista, Luís. (2021). Predict students' dropout and academic success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.

Dataset berisi 37 kolom dan 4424 baris, termasuk fitur seperti status perkawinan, mode aplikasi, kursus, nilai kualifikasi sebelumnya, dan status siswa (Dropout atau Graduate). Sel ini juga menampilkan 5 baris pertama dataset menggunakan df.head() dan ukuran dataset dengan df.shape. Penanganan kesalahan seperti file tidak ditemukan atau format file salah juga disertakan. Output menunjukkan dataset berhasil dimuat dengan 5 baris pertama dan dimensi dataset.

## Data exploration
![ss4](https://github.com/user-attachments/assets/cfe56384-9f65-4936-9617-37c06512b5b6)
![ss1](https://github.com/user-attachments/assets/fbe38e9d-f953-4aff-a0b2-10dca8f9150b)
# Analisis Awal Dataset

Sel ini berisi kode untuk melakukan analisis awal terhadap dataset, dengan tujuan memahami struktur data, mendeteksi nilai yang hilang, serta menganalisis distribusi fitur numerik dan kategorikal. Selain itu, sel ini juga menghitung matriks korelasi antar fitur numerik dan menampilkan visualisasinya dalam bentuk *heatmap*.

---

## Langkah-langkah Analisis

### 1. Impor Library

Langkah pertama adalah mengimpor *library* Python yang diperlukan:
* **`pandas`**: Digunakan untuk manipulasi dan analisis data.
* **`matplotlib.pyplot`** dan **`seaborn`**: Digunakan untuk visualisasi data.
* **`math`**: Digunakan untuk perhitungan matematis, seperti pembulatan untuk menentukan jumlah baris pada *subplot*.

### 2. Menampilkan Informasi DataFrame

* `df.info()`: Digunakan untuk menampilkan informasi dasar tentang dataset, seperti jumlah baris, kolom, tipe data setiap kolom, dan apakah ada nilai yang hilang. (Output spesifik tidak disertakan dalam dokumen ini).

### 3. Memeriksa Nilai yang Hilang

* `df.isnull().sum()`: Menghitung jumlah nilai yang hilang untuk setiap kolom.
* Persentase nilai yang hilang dihitung dengan `(missing_values / len(df)) * 100`.
* **Hasil**: Dataset ini **tidak memiliki nilai yang hilang**. (Output spesifik tidak disertakan dalam dokumen ini).

### 4. Ringkasan Statistik

* `df.describe()`: Menghasilkan statistik deskriptif (seperti rata-rata, standar deviasi, nilai minimum, maksimum, dan kuartil) untuk kolom numerik. (Output spesifik tidak disertakan dalam dokumen ini).

### 5. Distribusi Fitur Numerik

* Kolom numerik diidentifikasi menggunakan `df.select_dtypes(include=['number'])`.
* Histogram untuk setiap kolom numerik dibuat menggunakan `sns.histplot` dengan tambahan kurva KDE (Kernel Density Estimation) untuk melihat distribusi data.
* Visualisasi dilakukan dalam bentuk *subplot* dengan *grid* yang diatur secara dinamis (5 kolom per baris, baris disesuaikan dengan jumlah fitur). (Output visualisasi tidak disertakan dalam dokumen ini).

### 6. Analisis Fitur Kategorikal

* Kolom kategorikal diidentifikasi menggunakan `df.select_dtypes(include=['object', 'category'])`.
* Untuk setiap kolom kategorikal, `df[col].value_counts()` digunakan untuk menghitung frekuensi setiap kategori.
* **Output (Distribusi Kategori pada Kolom 'Status'):**
    * `Graduate`: 2209 siswa
    * `Dropout`: 1421 siswa
    * `Enrolled`: 794 siswa
* **Temuan**: Terdapat **ketidakseimbangan kelas**, dengan kategori 'Graduate' memiliki jumlah terbanyak, diikuti oleh 'Dropout', dan 'Enrolled' yang paling sedikit.

### 7. Matriks Korelasi dan Heatmap

* **Matriks korelasi** dihitung menggunakan `df.corr(numeric_only=True)` untuk fitur numerik saja.
* **Heatmap** dibuat menggunakan `sns.heatmap` dengan parameter:
    * `annot=True`: Menampilkan nilai korelasi pada setiap sel.
    * `cmap='coolwarm'`: Skema warna dari biru (korelasi negatif) hingga merah (korelasi positif).
    * `fmt=".2f"`: Format dua desimal.
    * Ukuran *figure* diatur ke `(15, 10)` untuk memastikan *heatmap* terlihat jelas.
* **Hasil**: *Heatmap* menunjukkan korelasi antar fitur numerik. (Visualisasi *heatmap* akan ditampilkan sebagai gambar terpisah).

## Data Cleaning
Berfokus pada pembersihan data dengan dua langkah utama: penanganan nilai yang hilang dan pengelolaan outlier pada dataset. Berikut adalah rincian proses yang dilakukan:

1. **Identifikasi Nilai yang Hilang**:  
   Menggunakan `df.isnull().sum()` untuk memeriksa jumlah nilai yang hilang di setiap kolom. Hasilnya menunjukkan bahwa tidak ada nilai yang hilang di semua kolom (semua bernilai 0), sehingga dataset sudah lengkap.

2. **Penanganan Nilai yang Hilang (Jika Ada)**:  
   - Untuk kolom numerik, nilai yang hilang akan diisi dengan **median** menggunakan `df[col].fillna(df[col].median(), inplace=True)`. Namun, karena tidak ada nilai yang hilang, langkah ini tidak dijalankan.
   - Untuk kolom kategorikal, nilai yang hilang akan diisi dengan **modus** (kategori yang paling sering muncul) menggunakan `df[col].fillna(df[col].mode()[0], inplace=True)`. Langkah ini juga tidak dijalankan karena tidak ada nilai yang hilang.

3. **Penanganan Outlier dengan Metode IQR**:  
   Outlier pada kolom numerik diidentifikasi dan ditangani menggunakan metode **Interquartile Range (IQR)**:  
   - Menghitung kuartil pertama (Q1) dan kuartil ketiga (Q3) untuk setiap kolom numerik.  
   - Menentukan batas bawah (`Q1 - 1.5 * IQR`) dan batas atas (`Q3 + 1.5 * IQR`).  
   - Nilai di luar batas ini dipotong (clipped) ke batas bawah atau atas menggunakan `df[col].clip(lower_bound, upper_bound)` untuk mengurangi dampak outlier tanpa menghapus data.

4. **Verifikasi Pembersihan Data**:  
   - Setelah penanganan, nilai yang hilang diperiksa kembali menggunakan `df.isnull().sum()`, yang mengkonfirmasi bahwa dataset tetap tidak memiliki nilai yang hilang.  
   - Statistik ringkasan fitur numerik ditampilkan menggunakan `df.describe()` untuk memeriksa distribusi data setelah penanganan outlier. Hasilnya menunjukkan bahwa beberapa kolom, seperti `Marital_status`, `Daytime_evening_attendance`, dan lainnya, memiliki deviasi standar 0, menunjukkan nilai konstan atau distribusi yang sangat terbatas setelah clipping. Kolom lain, seperti `Previous_qualification_grade`, `Curricular_units_2nd_sem_grade`, dan `Unemployment_rate`, menunjukkan rentang nilai yang telah disesuaikan untuk menghilangkan outlier ekstrem.

**Hasil Utama**:  
- Dataset tidak memiliki nilai yang hilang sejak awal, sehingga imputasi tidak diperlukan.  
- Outlier pada fitur numerik telah ditangani dengan metode IQR, memastikan distribusi data lebih stabil untuk pelatihan model.  
- Statistik ringkasan menunjukkan bahwa beberapa fitur memiliki variasi terbatas (misalnya, `Marital_status` dan `Nacionality` bernilai konstan), yang mungkin memengaruhi pentingnya fitur dalam model.  
- Data kini lebih bersih dan siap untuk langkah preprocessing berikutnya, seperti encoding dan pembuatan fitur baru.

## Data Preparation
Tahap ini menangani pengodean fitur kategoris dalam dataset untuk mempersiapkannya untuk pelatihan model machine learning. Langkah-langkah yang dilakukan adalah sebagai berikut:
1. **Identifikasi Kolom Kategoris**: Menggunakan `df.select_dtypes(exclude=['number'])` untuk mengidentifikasi kolom dengan tipe data non-numerik (kategoris), kecuali kolom `Status` yang merupakan target variabel dan dikecualikan dari daftar kolom kategoris.
2. **Pengodean Berdasarkan Kardinalitas**:
   - Untuk kolom kategoris dengan kardinalitas rendah (<10 nilai unik), diterapkan **one-hot encoding** menggunakan `pd.get_dummies()` dengan parameter `drop='first'` untuk menghindari multikolinearitas. Kolom asli kemudian dihapus, dan kolom baru hasil one-hot encoding digabungkan ke dataset.
   - Untuk kolom kategoris dengan kardinalitas tinggi (≥10 nilai unik), diterapkan **label encoding** menggunakan `sklearn.preprocessing.LabelEncoder` untuk mengubah nilai kategoris menjadi numerik.
3. **Penanganan Kesalahan**: Sel ini mencakup penanganan kesalahan selama proses pengodean label dengan menggunakan blok `try-except` untuk menangkap dan melaporkan potensi error.
4. **Verifikasi**: Setelah pengodean, informasi dataset diperiksa menggunakan `df.info()` untuk memastikan semua kolom telah diubah menjadi tipe data numerik (kecuali kolom `Status` yang tetap bertipe `object`).

**Hasil**:  
Output dari `df.info()` menunjukkan bahwa dataset memiliki 4.424 baris dan 37 kolom, dengan 36 kolom bertipe numerik (`int64` atau `float64`) dan 1 kolom (`Status`) bertipe `object`. Tidak ada nilai hilang (non-null count = 4.424 untuk semua kolom). Proses pengodean berhasil mengubah fitur kategoris menjadi format numerik yang sesuai untuk pelatihan model machine learning.

## Feature engineering
Tahap ini melakukan proses feature engineering dan penskalaan data untuk mempersiapkan dataset sebelum pelatihan model. Langkah-langkah yang dilakukan adalah:

Fitur Interaksi: Membuat dua fitur baru yaitu Interaction_Grade_Failed_Courses (hasil perkalian Curricular_units_1st_sem_grade dan Curricular_units_1st_sem_without_evaluations) dan Interaction_Mother_Father_Qual (hasil perkalian Mothers_qualification dan Fathers_qualification) untuk menangkap hubungan antar fitur.
Fitur Polinomial: Membuat fitur Age_Squared dengan mengkuadratkan Age_at_enrollment untuk menangkap hubungan non-linear.
Fitur Gabungan: Membuat fitur Academic_Performance_1st_Sem (rata-rata Curricular_units_1st_sem_grade dan Curricular_units_1st_sem_approved) dan Combined_Parents_Qual (rata-rata Mothers_qualification dan Fathers_qualification) untuk menggabungkan informasi terkait.
Penskalaan Data: Menggunakan RobustScaler dari sklearn.preprocessing untuk menskalakan semua kolom numerik agar tahan terhadap outlier. Scaler disimpan sebagai file robust_scaler.joblib untuk digunakan kembali.
Output: Menampilkan 5 baris pertama dari dataset yang telah diperbarui (df.head()) dan mengkonfirmasi bahwa scaler telah diekspor.
Hasil:

Scaler berhasil diekspor sebagai robust_scaler.joblib.
Dataset yang diperbarui menunjukkan fitur-fitur baru (Interaction_Grade_Failed_Courses, Interaction_Mother_Father_Qual, Age_Squared, Academic_Performance_1st_Sem, Combined_Parents_Qual) dan nilai-nilai numerik yang telah diskalakan menggunakan RobustScaler. Kolom Status tetap tidak diskalakan karena merupakan variabel target kategorikal.

## Data splitting
Tahap ini melakukan langkah awal dalam persiapan data untuk pelatihan model machine learning. Langkah-langkah yang dilakukan adalah sebagai berikut:

Encoding Label Target: Kolom target Status (yang berisi label kategorikal seperti "Dropout" atau "Graduate") diubah menjadi nilai numerik menggunakan LabelEncoder dari sklearn.preprocessing. Encoder ini disimpan ke file encoder_target.joblib menggunakan joblib untuk digunakan kembali saat prediksi di masa depan.

Pemisahan Fitur dan Target: Dataset dibagi menjadi fitur (X) dan target (y). Kolom Status dijadikan target (y), sedangkan semua kolom lain menjadi fitur (X).

Pembagian Data: Dataset dibagi menjadi tiga set: pelatihan (80%), validasi (10%), dan pengujian (10%).

Pertama, data dipisah menjadi set pelatihan (X_train, y_train) dan set sementara (X_temp, y_temp) dengan rasio 80:20 menggunakan train_test_split dengan parameter stratify=y untuk menjaga distribusi kelas.
Kemudian, set sementara dibagi lagi menjadi set validasi (X_val, y_val) dan set pengujian (X_test, y_test) dengan rasio 50:50 (masing-masing 10% dari total data).
Verifikasi Pembagian: Bentuk (shape) dari masing-masing set dicetak untuk memastikan pembagian data dilakukan dengan benar. Selain itu, distribusi kelas pada setiap set (y_train, y_val, y_test) diperiksa menggunakan value_counts(normalize=True) untuk memastikan proporsi kelas tetap konsisten di semua set.

Hasil:

Encoder target berhasil disimpan sebagai encoder_target.joblib.
Data dibagi dengan benar:
Set pelatihan: 3539 baris, 41 fitur.
Set validasi: 442 baris, 41 fitur.
Set pengujian: 443 baris, 41 fitur.
Distribusi kelas pada ketiga set menunjukkan proporsi yang hampir sama:
Kelas 2 (49.9–50%), Kelas 0 (32.1%), Kelas 1 (~17.9–18.1%).
Ini menunjukkan bahwa stratifikasi berhasil menjaga distribusi kelas yang seimbang di semua set.
Catatan: Konsistensi distribusi kelas penting untuk memastikan model dilatih dan dievaluasi pada data yang representatif, terutama karena adanya ketidakseimbangan kelas (Kelas 1 memiliki proporsi lebih kecil).

## Modelling (Model training)
Kode dimulai dengan mengimpor modul DecisionTreeClassifier, RandomForestClassifier, dan accuracy_score dari Scikit-learn untuk keperluan klasifikasi dan evaluasi. Kemudian, sebuah model Decision Tree diinisialisasi dan dilatih menggunakan data pelatihan (X_train dan y_train). Setelah itu, sebuah model Random Forest juga diinisialisasi dan dilatih menggunakan data pelatihan yang sama. Kedua model tersebut selanjutnya digunakan untuk membuat prediksi pada data validasi (X_val). Terakhir, akurasi prediksi dari masing-masing model dihitung dengan membandingkan hasil prediksi dengan label sebenarnya (y_val) dan kemudian menampilkan skor akurasi tersebut untuk kedua model.

## Model optimization
Tahap ini melakukan optimasi hyperparameter untuk dua model machine learning, yaitu Decision Tree dan Random Forest, menggunakan `GridSearchCV` dari library `sklearn`. Proses ini bertujuan untuk menemukan kombinasi parameter terbaik yang memaksimalkan akurasi model pada data pelatihan. Berikut adalah rincian langkah-langkah yang dilakukan:

1. **Impor Library**: Mengimpor `GridSearchCV`, `DecisionTreeClassifier`, dan `RandomForestClassifier` dari `sklearn` untuk keperluan optimasi dan pembuatan model.
2. **Definisi Parameter Grid**:
   - Untuk **Decision Tree**, parameter yang diuji meliputi:
     - `max_depth`: [None, 10, 20, 30] (kedalaman maksimum pohon).
     - `min_samples_split`: [2, 5, 10] (jumlah minimum sampel untuk memisahkan node).
     - `min_samples_leaf`: [1, 2, 4] (jumlah minimum sampel di node daun).
   - Untuk **Random Forest**, parameter yang diuji meliputi:
     - `n_estimators`: [100, 200, 300] (jumlah pohon dalam hutan).
     - `max_depth`: [None, 10, 20] (kedalaman maksimum pohon).
     - `min_samples_split`: [2, 5] (jumlah minimum sampel untuk memisahkan node).
     - `min_samples_leaf`: [1, 2] (jumlah minimum sampel di node daun).
     - `max_features`: ['sqrt', 'log2'] (jumlah fitur maksimum yang dipertimbangkan untuk pemisahan).
3. **Inisialisasi GridSearchCV**:
   - Untuk Decision Tree, `GridSearchCV` diinisialisasi dengan `DecisionTreeClassifier` (dengan `random_state=42` untuk reproduktifitas), parameter grid (`param_grid_dt`), 5-fold cross-validation, dan metrik evaluasi akurasi.
   - Untuk Random Forest, `GridSearchCV` diinisialisasi dengan `RandomForestClassifier` (dengan `random_state=42`), parameter grid (`param_grid_rf`), 5-fold cross-validation, dan metrik evaluasi akurasi.
4. **Pelatihan Model**:
   - GridSearchCV melatih model Decision Tree dan Random Forest pada data pelatihan (`X_train`, `y_train`) untuk semua kombinasi parameter yang ditentukan, menggunakan 5-fold cross-validation untuk mengevaluasi performa.
5. **Hasil Parameter Terbaik**:
   - Untuk Decision Tree, parameter terbaik adalah: `{'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10}`.
   - Untuk Random Forest, parameter terbaik adalah: `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 300}`.
6. **Output**: Parameter terbaik untuk kedua model dicetak, dan estimator terbaik (`best_dt_classifier` dan `best_rf_classifier`) disimpan untuk digunakan dalam langkah evaluasi atau prediksi selanjutnya.

**Hasil**:  
- **Decision Tree**: Parameter optimal (`max_depth=10`, `min_samples_leaf=4`, `min_samples_split=10`) menunjukkan model yang tidak terlalu dalam untuk menghindari overfitting, dengan batasan jumlah sampel di node untuk meningkatkan generalisasi.
- **Random Forest**: Parameter optimal (`max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=2`, `min_samples_split=2`, `n_estimators=300`) menunjukkan bahwa model memanfaatkan banyak pohon (300) dengan fitur acak (`sqrt`) dan batasan kecil pada node untuk menjaga fleksibilitas dan akurasi.

## Evaluation
![ss2](https://github.com/user-attachments/assets/026a99a8-9b18-4c4c-a729-a690740cc4f4)
![ss3](https://github.com/user-attachments/assets/f788b416-1709-47d1-83a1-c0ba219457d8)

Tahap ini melakukan evaluasi performa model Decision Tree dan Random Forest yang telah dilatih sebelumnya pada data uji (`X_test`, `y_test`). Proses ini mencakup perhitungan berbagai metrik evaluasi dan visualisasi matriks konfusi untuk memahami kekuatan dan kelemahan masing-masing model. Berikut adalah rincian langkah-langkah yang dilakukan:

1. **Impor Library**: Mengimpor metrik evaluasi seperti `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `classification_report`, dan `confusion_matrix` dari `sklearn.metrics`. Selain itu, `seaborn` dan `matplotlib.pyplot` diimpor untuk visualisasi.
2. **Prediksi**: Model terbaik (`best_dt_classifier` dan `best_rf_classifier`) digunakan untuk membuat prediksi pada data uji (`X_test`), menghasilkan `dt_predictions` dan `rf_predictions`.
3. **Fungsi Evaluasi**: Fungsi `evaluate_model` didefinisikan untuk menghitung dan menampilkan metrik evaluasi, termasuk:
   - Akurasi
   - Presisi (weighted average)
   - Recall (weighted average)
   - F1-score (weighted average)
   - AUC-ROC (jika berlaku, dengan penanganan khusus untuk data multikelas)
   - Laporan klasifikasi (`classification_report`) untuk detail per kelas
   - Visualisasi matriks konfusi menggunakan heatmap dari `seaborn`.
4. **Visualisasi Matriks Konfusi**: Matriks konfusi untuk kedua model ditampilkan dengan anotasi jumlah prediksi untuk setiap kombinasi aktual dan prediksi, menggunakan skema warna biru (`Blues`).
5. **Perbandingan Model**: Metrik utama (akurasi dan F1-score) dari kedua model dibandingkan untuk menentukan model mana yang lebih baik secara keseluruhan.
6. **Output**:
   - **Decision Tree Metrics**:
     - Akurasi: 0.6659
     - Presisi: 0.6640
     - Recall: 0.6659
     - F1-score: 0.6649
     - AUC-ROC: Tidak berlaku untuk multikelas
     - Laporan klasifikasi menunjukkan performa per kelas: Dropout (0.67/0.65), Enrolled (0.34/0.34), Graduate (0.77/0.79).
     - Matriks konfusi: 93 (0,0), 27 (0,1), 22 (0,2), 24 (1,0), 27 (1,1), 29 (1,2), 21 (2,0), 25 (2,1), 175 (2,2).
   - **Random Forest Metrics**:
     - Akurasi: 0.7607
     - Presisi: 0.7443
     - Recall: 0.7607
     - F1-score: 0.7436
     - AUC-ROC: Tidak berlaku untuk multikelas
     - Laporan klasifikasi menunjukkan performa per kelas: Dropout (0.77/0.73), Enrolled (0.57/0.34), Graduate (0.79/0.94).
     - Matriks konfusi: 103 (0,0), 11 (0,1), 28 (0,2), 26 (1,0), 27 (1,1), 27 (1,2), 5 (2,0), 9 (2,1), 207 (2,2).
   - **Perbandingan Model**:
     - Decision Tree: Akurasi 0.6659, Presisi 0.6640, Recall 0.6659, F1-score 0.6649
     - Random Forest: Akurasi 0.7607, Presisi 0.7443, Recall 0.7607, F1-score 0.7436
     - Kesimpulan: Random Forest berkinerja lebih baik daripada Decision Tree dalam hal akurasi dan skor F1.

**Hasil**:  
- Random Forest menunjukkan performa superior dibandingkan Decision Tree, dengan akurasi 76.07% versus 66.59% dan F1-score 0.7436 versus 0.6649. Matriks konfusi menunjukkan Random Forest lebih akurat dalam memprediksi kelas "Graduate" (207 benar) dibandingkan Decision Tree (175 benar), meskipun keduanya memiliki kesulitan dengan kelas "Enrolled". Hal ini menegaskan bahwa Random Forest adalah pilihan yang lebih baik untuk tugas prediksi dropout ini.

## Ekspor Model
Tahap ini bertujuan untuk menyimpan model machine learning yang telah dioptimalkan serta dataset yang telah dibersihkan untuk penggunaan lebih lanjut, seperti implementasi pada dashboard Streamlit. Berikut adalah rincian langkah-langkah yang dilakukan:

Impor Library: Mengimpor joblib untuk menyimpan model machine learning ke dalam file eksternal.
Ekspor Model Decision Tree: Model Decision Tree terbaik (best_dt_classifier), yang telah dioptimalkan menggunakan GridSearchCV, disimpan ke file decision_tree_model.joblib menggunakan fungsi joblib.dump.
Ekspor Model Random Forest: Model Random Forest terbaik (best_rf_classifier), yang juga telah dioptimalkan, disimpan ke file random_forest_model.joblib menggunakan fungsi joblib.dump.
Ekspor Dataset: Dataset yang telah dibersihkan dan diproses (df) diekspor ke file CSV bernama cleaned_student_data.csv menggunakan df.to_csv dengan parameter index=False untuk menghilangkan indeks baris dalam file output.
Output: Pesan konfirmasi dicetak untuk menginformasikan bahwa model telah berhasil diekspor sebagai decision_tree_model.joblib dan random_forest_model.joblib, serta dataset telah diekspor sebagai cleaned_student_data.csv.
Hasil:

Model Decision Tree dan Random Forest berhasil disimpan sebagai file joblib, memungkinkan penggunaan kembali untuk prediksi pada aplikasi seperti dashboard Streamlit.
Dataset yang telah dibersihkan disimpan sebagai cleaned_student_data.csv, yang dapat digunakan untuk analisis lebih lanjut atau sebagai input untuk model pada sistem lain.

## Setup Environment :

1. Buat dan aktifkan virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate     # Untuk Windows
   ```
2. Instal dependensi dari `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Isi file `requirements.txt`:

   ```
   streamlit
   joblib
   pandas
   numpy
   scikit-learn
   ```
3. Jalankan aplikasi Streamlit:

   ```bash
   streamlit run app.py
   ```

## Business Dashboard

Dashboard dibuat menggunakan **Looker Studio** untuk memvisualisasikan data siswa dan hasil prediksi model machine learning. Dashboard ini menampilkan distribusi status siswa (Dropout, Enrolled, Graduate), performa akademik, serta faktor risiko dropout seperti nilai masuk, usia, dan kualifikasi orang tua. Fitur interaktif memungkinkan pengguna untuk memfilter data berdasarkan variabel tertentu untuk analisis mendalam.\
**Link Dashboard**: [https://lookerstudio.google.com/reporting/f2e3d846-942a-48ed-a75d-ae3f8a036b3b]
![image](https://github.com/user-attachments/assets/b395ff9e-57e9-434b-841e-b26fcfcb740a)


## Menjalankan Sistem Machine Learning

Prototipe sistem machine learning dibuat menggunakan **Streamlit** untuk memprediksi risiko dropout siswa secara real-time. Langkah-langkah menjalankan sistem:

1. Pastikan virtual environment telah diatur dan dependensi terinstal (lihat bagian Setup Environment).
2. Unduh file model (`decision_tree_model.joblib`, `random_forest_model.joblib`) dan dataset (`cleaned_student_data.csv`) dari notebook.
3. Jalankan aplikasi Streamlit dengan perintah:

   ```bash
   streamlit run app.py
   ```
4. Masukkan data siswa (seperti usia, nilai masuk, performa akademik, dll.) melalui antarmuka Streamlit untuk mendapatkan prediksi status (Dropout, Enrolled, atau Graduate). **Link Prototipe**: [https://penerapan-data-science-2-eldy.streamlit.app/]
![image](https://github.com/user-attachments/assets/7f7173c4-ef5c-4c62-82ce-d7ba4263a35c)
![image](https://github.com/user-attachments/assets/fbc0acad-fb4c-4bdf-b922-464b396ded59)

## Conclusion

- **Performa Model**: Model Random Forest mengungguli Decision Tree dengan akurasi 76.07% (vs. 66.59%) dan F1-score 0.7436 (vs. 0.6649). Random Forest lebih akurat dalam memprediksi kelas "Graduate" dan lebih robust secara keseluruhan.
- **Temuan Data**: Dataset tidak memiliki nilai hilang, dan outlier ditangani dengan metode IQR. Fitur baru seperti `Interaction_Grade_Failed_Courses` meningkatkan performa model. Kelas "Enrolled" sulit diprediksi karena ketidakseimbangan data.
- **Manfaat Bisnis**: Sistem ini memungkinkan Jaya Jaya Institut untuk mengidentifikasi siswa berisiko dropout secara dini, memungkinkan intervensi tepat waktu dan optimalisasi sumber daya bimbingan.
- **Implementasi**: Model dan data telah diekspor untuk digunakan dalam aplikasi Streamlit, dan dashboard Looker Studio memberikan wawasan visual yang mendukung pengambilan keputusan.

## Rekomendasi Action Items

1. **Implementasi Sistem Prediksi di Institusi**: Integrasikan aplikasi Streamlit ke dalam sistem manajemen siswa Jaya Jaya Institut untuk memprediksi risiko dropout secara real-time dan memberikan peringatan kepada staf akademik untuk tindakan bimbingan.
2. **Program Bimbingan Khusus**: Kembangkan program bimbingan berbasis hasil prediksi, fokus pada siswa dengan risiko tinggi (misalnya, berdasarkan performa akademik rendah atau faktor sosial-ekonomi) untuk meningkatkan retensi siswa.
3. **Penanganan Ketidakseimbangan Kelas**: Terapkan teknik seperti oversampling (SMOTE) atau undersampling pada kelas "Enrolled" untuk meningkatkan performa model pada kelas minoritas.

## Pemantauan dan Pembaruan Model : 
Lakukan pembaruan berkala pada model dengan data siswa baru untuk memastikan akurasi prediksi tetap relevan, serta evaluasi fitur baru yang mungkin memengaruhi dropout.
