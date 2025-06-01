import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Streamlit App Configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Jaya Jaya Institut Dropout Prediction", layout="wide")

# --- Configuration ---
MODEL_PATH = 'model/random_forest_model.joblib'
DATA_PATH = 'dataset/cleaned_student_data.csv'

# --- Load Model and Data ---
@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Please make sure 'random_forest_model.joblib' is in the 'model/' directory.")
        return None

@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset file not found at {path}. Please make sure 'cleaned_student_data.csv' is in the 'dataset/' directory.")
        return None

model = load_model(MODEL_PATH)
df_template = load_data(DATA_PATH)

if model is None or df_template is None:
    st.stop()

# Drop the 'Status' column from the template DataFrame as it's the target
if 'Status' in df_template.columns:
    df_template = df_template.drop(columns=['Status'])

# --- Mapping Dictionaries for User-Friendly Options ---
MARITAL_STATUS = {
    "Single": 1,
    "Married": 2,
    "Widower": 3,
    "Divorced": 4,
    "Facto Union": 5,
    "Legally Separated": 6
}

APPLICATION_MODE = {
    "1st phase - general contingent": 1,
    "Ordinance No. 612/93": 2,
    "1st phase - special contingent (Azores Island)": 5,
    "Holders of other higher courses": 7,
    "Ordinance No. 854-B/99": 10,
    "International student (bachelor)": 15,
    "1st phase - special contingent (Madeira Island)": 16,
    "2nd phase - general contingent": 17,
    "3rd phase - general contingent": 18,
    "Ordinance No. 533-A/99, item b2) (Different Plan)": 26,
    "Ordinance No. 533-A/99, item b3 (Other Institution)": 27,
    "Over 23 years old": 39,
    "Transfer": 42,
    "Change of course": 43,
    "Technological specialization diploma holders": 44,
    "Change of institution/course": 51,
    "Short cycle diploma holders": 53,
    "Change of institution/course (International)": 57
}

COURSES = {
    "Biofuel Production Technologies": 33,
    "Animation and Multimedia Design": 171,
    "Social Service (evening attendance)": 8014,
    "Agronomy": 9003,
    "Communication Design": 9070,
    "Veterinary Nursing": 9085,
    "Informatics Engineering": 9119,
    "Equinculture": 9130,
    "Management": 9147,
    "Social Service": 9238,
    "Tourism": 9254,
    "Nursing": 9500,
    "Oral Hygiene": 9556,
    "Advertising and Marketing Management": 9670,
    "Journalism and Communication": 9773,
    "Basic Education": 9853,
    "Management (evening attendance)": 9991
}

PREVIOUS_QUALIFICATION = {
    "Secondary education": 1,
    "Higher education - bachelor's degree": 2,
    "Higher education - degree": 3,
    "Higher education - master's": 4,
    "Higher education - doctorate": 5,
    "Frequency of higher education": 6,
    "12th year of schooling - not completed": 9,
    "11th year of schooling - not completed": 10,
    "Other - 11th year of schooling": 12,
    "10th year of schooling": 14,
    "10th year of schooling - not completed": 15,
    "Basic education 3rd cycle (9th/10th/11th year) or equiv.": 19,
    "Basic education 2nd cycle (6th/7th/8th year) or equiv.": 38,
    "Technological specialization course": 39,
    "Higher education - degree (1st cycle)": 40,
    "Professional higher technical course": 42,
    "Higher education - master (2nd cycle)": 43
}

NATIONALITY = {
    "Portuguese": 1,
    "German": 2,
    "Spanish": 6,
    "Italian": 11,
    "Dutch": 13,
    "English": 14,
    "Lithuanian": 17,
    "Angolan": 21,
    "Cape Verdean": 22,
    "Guinean": 24,
    "Mozambican": 25,
    "Santomean": 26,
    "Turkish": 32,
    "Brazilian": 41,
    "Romanian": 62,
    "Moldova (Republic of)": 100,
    "Mexican": 101,
    "Ukrainian": 103,
    "Russian": 105,
    "Cuban": 108,
    "Colombian": 109
}

PARENT_QUALIFICATION = {
    "Secondary Education - 12th Year of Schooling or Eq.": 1,
    "Higher Education - Bachelor's Degree": 2,
    "Higher Education - Degree": 3,
    "Higher Education - Master's": 4,
    "Higher Education - Doctorate": 5,
    "Frequency of Higher Education": 6,
    "12th Year of Schooling - Not Completed": 9,
    "11th Year of Schooling - Not Completed": 10,
    "7th Year (Old)": 11,
    "Other - 11th Year of Schooling": 12,
    "2nd year complementary high school course": 13,
    "10th Year of Schooling": 14,
    "General commerce course": 18,
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.": 19,
    "Complementary High School Course": 20,
    "Technical-professional course": 22,
    "Complementary High School Course - not concluded": 25,
    "7th year of schooling": 26,
    "2nd cycle of the general high school course": 27,
    "9th Year of Schooling - Not Completed": 29,
    "8th year of schooling": 30,
    "General Course of Administration and Commerce": 31,
    "Supplementary Accounting and Administration": 33,
    "Unknown": 34,
    "Can't read or write": 35,
    "Can read without having a 4th year of schooling": 36,
    "Basic education 1st cycle (4th/5th year) or equiv.": 37,
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.": 38,
    "Technological specialization course": 39,
    "Higher education - degree (1st cycle)": 40,
    "Specialized higher studies course": 41,
    "Professional higher technical course": 42,
    "Higher Education - Master (2nd cycle)": 43,
    "Higher Education - Doctorate (3rd cycle)": 44
}

PARENT_OCCUPATION = {
    "Student": 0,
    "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers": 1,
    "Specialists in Intellectual and Scientific Activities": 2,
    "Intermediate Level Technicians and Professions": 3,
    "Administrative staff": 4,
    "Personal Services, Security and Safety Workers and Sellers": 5,
    "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry": 6,
    "Skilled Workers in Industry, Construction and Craftsmen": 7,
    "Installation and Machine Operators and Assembly Workers": 8,
    "Unskilled Workers": 9,
    "Armed Forces Professions": 10,
    "Other Situation": 90,
    "(blank)": 99,
    "Health professionals": 122,
    "Teachers": 123,
    "Specialists in information and communication technologies (ICT)": 125,
    "Intermediate level science and engineering technicians and professions": 131,
    "Technicians and professionals, of intermediate level of health": 132,
    "Intermediate level technicians from legal, social, sports, cultural and similar services": 134,
    "Office workers, secretaries in general and data processing operators": 141,
    "Data, accounting, statistical, financial services and registry-related operators": 143,
    "Other administrative support staff": 144,
    "Personal service workers": 151,
    "Sellers": 152,
    "Personal care workers and the like": 153,
    "Skilled construction workers and the like, except electricians": 171,
    "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like": 173,
    "Workers in food processing, woodworking, clothing and other industries and crafts": 175,
    "Cleaning workers": 191,
    "Unskilled workers in agriculture, animal production, fisheries and forestry": 192,
    "Unskilled workers in extractive industry, construction, manufacturing and transport": 193,
    "Meal preparation assistants": 194
}

# --- Streamlit App Content ---
st.title("Jaya Jaya Institut: Deteksi Dini Potensi Dropout")
st.markdown("""
Aplikasi ini membantu Jaya Jaya Institut untuk memprediksi siswa yang kemungkinan akan dropout
berdasarkan berbagai informasi penting. Dengan deteksi dini, bimbingan khusus dapat diberikan.
""")

st.header("Informasi Lengkap Siswa")

# Prepare a dictionary to hold all user inputs
user_inputs = {}

# Group inputs into sections using expanders
# --- Bagian 1: Informasi Pribadi & Pendaftaran ---
with st.expander("1. Informasi Pribadi & Pendaftaran", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        user_inputs['Marital_status'] = st.selectbox(
            "Status Pernikahan", options=list(MARITAL_STATUS.keys()),
            help="Status pernikahan siswa saat ini."
        )
        user_inputs['Application_mode'] = st.selectbox(
            "Mode Aplikasi", options=list(APPLICATION_MODE.keys()),
            help="Cara siswa mendaftar ke institusi (misalnya, tes masuk, jalur prestasi)."
        )
        user_inputs['Application_order'] = st.selectbox(
            "Urutan Aplikasi", list(range(10)), index=0,
            help="Urutan aplikasi siswa (0 - pilihan pertama, 9 - pilihan terakhir)."
        )
    with col2:
        user_inputs['Course'] = st.selectbox(
            "Jurusan/Program Studi", options=list(COURSES.keys()),
            help="Program studi yang diambil siswa."
        )
        user_inputs['Daytime_evening_attendance'] = st.selectbox(
            "Kehadiran Siang/Malam", options=["Siang", "Malam"], index=0,
            help="Apakah siswa kuliah di kelas siang atau malam."
        )
        user_inputs['Previous_qualification'] = st.selectbox(
            "Kualifikasi Sebelumnya", options=list(PREVIOUS_QUALIFICATION.keys()),
            help="Kualifikasi pendidikan terakhir sebelum masuk institusi (misalnya, SMA, D3)."
        )
    with col3:
        user_inputs['Previous_qualification_grade'] = st.number_input(
            "Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=120.0,
            help="Nilai rata-rata kualifikasi pendidikan sebelumnya."
        )
        user_inputs['Nacionality'] = st.selectbox(
            "Kewarganegaraan", options=list(NATIONALITY.keys()),
            help="Kewarganegaraan siswa."
        )
        user_inputs['Admission_grade'] = st.number_input(
            "Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=120.0,
            help="Nilai rata-rata siswa saat masuk ke Jaya Jaya Institut."
        )

# --- Bagian 2: Latar Belakang Keluarga & Demografi ---
with st.expander("2. Latar Belakang Keluarga & Demografi"):
    col1, col2, col3 = st.columns(3)
    with col1:
        user_inputs['Mothers_qualification'] = st.selectbox(
            "Kualifikasi Pendidikan Ibu", options=list(PARENT_QUALIFICATION.keys()),
            help="Tingkat pendidikan ibu siswa."
        )
        user_inputs['Fathers_qualification'] = st.selectbox(
            "Kualifikasi Pendidikan Ayah", options=list(PARENT_QUALIFICATION.keys()),
            help="Tingkat pendidikan ayah siswa."
        )
        user_inputs['Mothers_occupation'] = st.selectbox(
            "Pekerjaan Ibu", options=list(PARENT_OCCUPATION.keys()),
            help="Jenis pekerjaan ibu siswa."
        )
    with col2:
        user_inputs['Fathers_occupation'] = st.selectbox(
            "Pekerjaan Ayah", options=list(PARENT_OCCUPATION.keys()),
            help="Jenis pekerjaan ayah siswa."
        )
        user_inputs['Gender'] = st.selectbox(
            "Jenis Kelamin", options=["Laki-laki", "Perempuan"], index=0,
            help="Jenis kelamin siswa."
        )
        user_inputs['Age_at_enrollment'] = st.number_input(
            "Usia Saat Pendaftaran (Tahun)", min_value=17, max_value=60, value=20,
            help="Usia siswa pada saat pertama kali mendaftar."
        )
    with col3:
        user_inputs['Displaced'] = st.selectbox(
            "Pindah Domisili", options=["Tidak", "Ya"], index=0,
            help="Apakah siswa pindah domisili untuk kuliah?"
        )
        user_inputs['International'] = st.selectbox(
            "Siswa Internasional", options=["Tidak", "Ya"], index=0,
            help="Apakah siswa berasal dari luar negeri?"
        )

# --- Bagian 3: Status Akademik & Finansial ---
with st.expander("3. Status Akademik & Finansial"):
    col1, col2, col3 = st.columns(3)
    with col1:
        user_inputs['Educational_special_needs'] = st.selectbox(
            "Kebutuhan Khusus Pendidikan", options=["Tidak", "Ya"], index=0,
            help="Apakah siswa memiliki kebutuhan khusus pendidikan?"
        )
        user_inputs['Debtor'] = st.selectbox(
            "Memiliki Tunggakan", options=["Tidak", "Ya"], index=0,
            help="Apakah siswa memiliki tunggakan pembayaran?"
        )
        user_inputs['Tuition_fees_up_to_date'] = st.selectbox(
            "Pembayaran Uang Kuliah Tepat Waktu", options=["Tidak", "Ya"], index=1,
            help="Apakah siswa membayar uang kuliah tepat waktu?"
        )
    with col2:
        user_inputs['Scholarship_holder'] = st.selectbox(
            "Penerima Beasiswa", options=["Tidak", "Ya"], index=0,
            help="Apakah siswa adalah penerima beasiswa?"
        )
        user_inputs['Curricular_units_1st_sem_credited'] = st.number_input(
            "Unit Kurikuler Sem 1 yang Dikreditkan", min_value=0, max_value=30, value=0,
            help="Jumlah unit kurikuler dari semester 1 yang dikreditkan."
        )
        user_inputs['Curricular_units_1st_sem_enrolled'] = st.number_input(
            "Unit Kurikuler Sem 1 yang Diambil", min_value=0, max_value=30, value=6,
            help="Jumlah unit kurikuler yang diambil di semester 1."
        )
    with col3:
        user_inputs['Curricular_units_1st_sem_evaluations'] = st.number_input(
            "Evaluasi Unit Kurikuler Sem 1", min_value=0, max_value=30, value=5,
            help="Jumlah evaluasi yang diterima untuk unit kurikuler di semester 1."
        )
        user_inputs['Curricular_units_1st_sem_approved'] = st.number_input(
            "Unit Kurikuler Sem 1 yang Lulus", min_value=0, max_value=30, value=5,
            help="Jumlah unit kurikuler yang berhasil diselesaikan/lulus di semester 1."
        )
        user_inputs['Curricular_units_1st_sem_grade'] = st.number_input(
            "Nilai Rata-rata Unit Kurikuler Sem 1", min_value=0.0, max_value=20.0, value=10.0,
            help="Nilai rata-rata unit kurikuler di semester 1."
        )
        user_inputs['Curricular_units_1st_sem_without_evaluations'] = st.number_input(
            "Unit Kurikuler Sem 1 Tanpa Evaluasi", min_value=0, max_value=30, value=0,
            help="Jumlah unit kurikuler di semester 1 yang tidak ada evaluasinya."
        )

# --- Bagian 4: Akademik Semester 2 ---
with st.expander("4. Akademik Semester 2"):
    col1, col2, col3 = st.columns(3)
    with col1:
        user_inputs['Curricular_units_2nd_sem_credited'] = st.number_input(
            "Unit Kurikuler Sem 2 yang Dikreditkan", min_value=0, max_value=30, value=0,
            help="Jumlah unit kurikuler dari semester 2 yang dikreditkan."
        )
        user_inputs['Curricular_units_2nd_sem_enrolled'] = st.number_input(
            "Unit Kurikuler Sem 2 yang Diambil", min_value=0, max_value=30, value=6,
            help="Jumlah unit kurikuler yang diambil di semester 2."
        )
    with col2:
        user_inputs['Curricular_units_2nd_sem_evaluations'] = st.number_input(
            "Evaluasi Unit Kurikuler Sem 2", min_value=0, max_value=30, value=5,
            help="Jumlah evaluasi yang diterima untuk unit kurikuler di semester 2."
        )
        user_inputs['Curricular_units_2nd_sem_approved'] = st.number_input(
            "Unit Kurikuler Sem 2 yang Lulus", min_value=0, max_value=30, value=5,
            help="Jumlah unit kurikuler yang berhasil diselesaikan/lulus di semester 2."
        )
    with col3:
        user_inputs['Curricular_units_2nd_sem_grade'] = st.number_input(
            "Nilai Rata-rata Unit Kurikuler Sem 2", min_value=0.0, max_value=20.0, value=10.0,
            help="Nilai rata-rata unit kurikuler di semester 2."
        )
        user_inputs['Curricular_units_2nd_sem_without_evaluations'] = st.number_input(
            "Unit Kurikuler Sem 2 Tanpa Evaluasi", min_value=0, max_value=30, value=0,
            help="Jumlah unit kurikuler di semester 2 yang tidak ada evaluasinya."
        )

# --- Bagian 5: Indikator Ekonomi Makro ---
with st.expander("5. Indikator Ekonomi Makro"):
    col1, col2, col3 = st.columns(3)
    with col1:
        user_inputs['Unemployment_rate'] = st.number_input(
            "Tingkat Pengangguran (%)", min_value=0.0, max_value=30.0, value=7.0, step=0.1,
            help="Tingkat pengangguran di wilayah studi pada tahun pendaftaran."
        )
    with col2:
        user_inputs['Inflation_rate'] = st.number_input(
            "Tingkat Inflasi (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.1,
            help="Tingkat inflasi di wilayah studi pada tahun pendaftaran."
        )
    with col3:
        user_inputs['GDP'] = st.number_input(
            "GDP (Gross Domestic Product)", min_value=0.0, max_value=200000.0, value=100000.0, step=100.0,
            help="Produk Domestik Bruto di wilayah studi pada tahun pendaftaran."
        )

# --- Prediction Button ---
st.markdown("---")
if st.button("Prediksi Potensi Dropout"):
    if model is not None and df_template is not None:
        # Convert user-friendly inputs back to numerical values
        user_inputs['Marital_status'] = MARITAL_STATUS[user_inputs['Marital_status']]
        user_inputs['Application_mode'] = APPLICATION_MODE[user_inputs['Application_mode']]
        user_inputs['Course'] = COURSES[user_inputs['Course']]
        user_inputs['Daytime_evening_attendance'] = 1 if user_inputs['Daytime_evening_attendance'] == "Siang" else 0
        user_inputs['Previous_qualification'] = PREVIOUS_QUALIFICATION[user_inputs['Previous_qualification']]
        user_inputs['Nacionality'] = NATIONALITY[user_inputs['Nacionality']]
        user_inputs['Mothers_qualification'] = PARENT_QUALIFICATION[user_inputs['Mothers_qualification']]
        user_inputs['Fathers_qualification'] = PARENT_QUALIFICATION[user_inputs['Fathers_qualification']]
        user_inputs['Mothers_occupation'] = PARENT_OCCUPATION[user_inputs['Mothers_occupation']]
        user_inputs['Fathers_occupation'] = PARENT_OCCUPATION[user_inputs['Fathers_occupation']]
        user_inputs['Gender'] = 1 if user_inputs['Gender'] == "Laki-laki" else 0
        user_inputs['Displaced'] = 1 if user_inputs['Displaced'] == "Ya" else 0
        user_inputs['International'] = 1 if user_inputs['International'] == "Ya" else 0
        user_inputs['Educational_special_needs'] = 1 if user_inputs['Educational_special_needs'] == "Ya" else 0
        user_inputs['Debtor'] = 1 if user_inputs['Debtor'] == "Ya" else 0
        user_inputs['Tuition_fees_up_to_date'] = 1 if user_inputs['Tuition_fees_up_to_date'] == "Ya" else 0
        user_inputs['Scholarship_holder'] = 1 if user_inputs['Scholarship_holder'] == "Ya" else 0

        # Create a new DataFrame for prediction
        input_df = pd.DataFrame(index=[0], columns=df_template.columns)

        # Fill with median/mode values from the template
        for col in df_template.columns:
            if col in ['Gender', 'Daytime_evening_attendance', 'Displaced', 'Educational_special_needs',
                       'Debtor', 'Tuition_fees_up_to_date', 'Scholarship_holder', 'International']:
                input_df[col] = df_template[col].mode()[0]
            elif df_template[col].dtype == 'object':
                input_df[col] = df_template[col].mode()[0]
            else:
                input_df[col] = df_template[col].median()

        # Overwrite with user inputs
        for key, value in user_inputs.items():
            if key in input_df.columns:
                input_df[key] = value

        # --- Re-create Engineered Features ---
        input_df['Age_Squared'] = input_df['Age_at_enrollment'] ** 2
        
        if input_df['Curricular_units_1st_sem_enrolled'].iloc[0] > 0:
            input_df['Academic_Performance_1st_Sem'] = (input_df['Curricular_units_1st_sem_grade'] * input_df['Curricular_units_1st_sem_approved']) / input_df['Curricular_units_1st_sem_enrolled']
        else:
            input_df['Academic_Performance_1st_Sem'] = 0.0

        failed_courses_1st_sem = input_df['Curricular_units_1st_sem_enrolled'] - input_df['Curricular_units_1st_sem_approved']
        failed_courses_2nd_sem = input_df['Curricular_units_2nd_sem_enrolled'] - input_df['Curricular_units_2nd_sem_approved']
        total_failed_courses = failed_courses_1st_sem + failed_courses_2nd_sem

        input_df['Interaction_Grade_Failed_Courses'] = input_df['Admission_grade'] * total_failed_courses
        input_df['Interaction_Mother_Father_Qual'] = input_df['Mothers_qualification'] * input_df['Fathers_qualification']
        input_df['Combined_Parents_Qual'] = input_df['Mothers_qualification'] + input_df['Fathers_qualification']

        input_data_processed = pd.DataFrame(0, index=[0], columns=df_template.columns)
        for col in input_df.columns:
            if col in input_data_processed.columns:
                input_data_processed[col] = input_df[col].iloc[0]
        
        input_data_processed = input_data_processed[df_template.columns]

        # Make prediction
        try:
            prediction = model.predict(input_data_processed)
            prediction_proba = model.predict_proba(input_data_processed)

            st.subheader("Hasil Prediksi")
            if prediction[0] == 1:
                st.error(f"Siswa ini **sangat berpotensi untuk DO (Dropout)** dengan probabilitas: **{prediction_proba[0][1]*100:.2f}%**")
                st.markdown("""
                    <div style='background-color: #ffe6e6; padding: 10px; border-radius: 5px; border: 1px solid #ffcccc;'>
                        **Rekomendasi:** Segera berikan bimbingan dan pendampingan khusus kepada siswa ini.
                        Pertimbangkan konseling akademik, dukungan finansial, atau program mentorship.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"Siswa ini **kemungkinan besar akan menyelesaikan pendidikannya** dengan probabilitas: **{prediction_proba[0][0]*100:.2f}%**")
                st.markdown("""
                    <div style='background-color: #e6ffe6; padding: 10px; border-radius: 5px; border: 1px solid #ccffcc;'>
                        **Rekomendasi:** Terus pantau perkembangan siswa, namun saat ini tidak ada indikasi kuat potensi dropout.
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Detail Input yang Digunakan:")
            st.write(input_data_processed)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.info("Pastikan format data yang dimasukkan sesuai dengan yang diharapkan model. Error: " + str(e))
    else:
        st.warning("Model atau data template tidak dapat dimuat. Silakan periksa kembali file Anda.")

st.markdown("---")
st.caption("Aplikasi Prediksi Dropout oleh Jaya Jaya Institut")
