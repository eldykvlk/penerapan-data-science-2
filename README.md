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


**Setup Environment**:

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

## Menjalankan Sistem Machine Learning

Prototipe sistem machine learning dibuat menggunakan **Streamlit** untuk memprediksi risiko dropout siswa secara real-time. Langkah-langkah menjalankan sistem:

1. Pastikan virtual environment telah diatur dan dependensi terinstal (lihat bagian Setup Environment).
2. Unduh file model (`decision_tree_model.joblib`, `random_forest_model.joblib`) dan dataset (`cleaned_student_data.csv`) dari notebook.
3. Jalankan aplikasi Streamlit dengan perintah:

   ```bash
   streamlit run app.py
   ```
4. Masukkan data siswa (seperti usia, nilai masuk, performa akademik, dll.) melalui antarmuka Streamlit untuk mendapatkan prediksi status (Dropout, Enrolled, atau Graduate). **Link Prototipe**: [https://penerapan-data-science-2-eldy.streamlit.app/]

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
