import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Streamlit App Configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Jaya Jaya Institut Dropout Prediction", layout="wide") # Changed to wide layout for more inputs

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
    st.stop() # Stop the app if crucial files are missing

# Drop the 'Status' column from the template DataFrame as it's the target
if 'Status' in df_template.columns:
    df_template = df_template.drop(columns=['Status'])

# --- Function to get unique values for selectboxes ---
def get_unique_values(column_name):
    if column_name in df_template.columns:
        return sorted(df_template[column_name].unique().tolist())
    return []

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
            "Status Pernikahan", get_unique_values('Marital_status'),
            help="Status pernikahan siswa saat ini."
        )
        user_inputs['Application_mode'] = st.selectbox(
            "Mode Aplikasi", get_unique_values('Application_mode'),
            help="Cara siswa mendaftar ke institusi (misalnya, tes masuk, jalur prestasi)."
        )
        user_inputs['Application_order'] = st.selectbox(
            "Urutan Aplikasi", list(range(10)), index=0,
            help="Urutan aplikasi siswa (0 - pilihan pertama, 9 - pilihan terakhir)."
        )
    with col2:
        user_inputs['Course'] = st.selectbox(
            "Jurusan/Program Studi", get_unique_values('Course'),
            help="Program studi yang diambil siswa."
        )
        user_inputs['Daytime_evening_attendance'] = st.selectbox(
            "Kehadiran Siang/Malam", options=[0, 1], format_func=lambda x: "Malam" if x==0 else "Siang", index=1,
            help="0: Malam, 1: Siang. Apakah siswa kuliah di kelas siang atau malam."
        )
        user_inputs['Previous_qualification'] = st.selectbox(
            "Kualifikasi Sebelumnya", get_unique_values('Previous_qualification'),
            help="Kualifikasi pendidikan terakhir sebelum masuk institusi (misalnya, SMA, D3)."
        )
    with col3:
        user_inputs['Previous_qualification_grade'] = st.number_input(
            "Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=120.0,
            help="Nilai rata-rata kualifikasi pendidikan sebelumnya."
        )
        user_inputs['Nacionality'] = st.selectbox(
            "Kewarganegaraan", get_unique_values('Nacionality'),
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
            "Kualifikasi Pendidikan Ibu", get_unique_values('Mothers_qualification'),
            help="Tingkat pendidikan ibu siswa."
        )
        user_inputs['Fathers_qualification'] = st.selectbox(
            "Kualifikasi Pendidikan Ayah", get_unique_values('Fathers_qualification'),
            help="Tingkat pendidikan ayah siswa."
        )
        user_inputs['Mothers_occupation'] = st.selectbox(
            "Pekerjaan Ibu", get_unique_values('Mothers_occupation'),
            help="Jenis pekerjaan ibu siswa."
        )
    with col2:
        user_inputs['Fathers_occupation'] = st.selectbox(
            "Pekerjaan Ayah", get_unique_values('Fathers_occupation'),
            help="Jenis pekerjaan ayah siswa."
        )
        user_inputs['Gender'] = st.selectbox(
            "Jenis Kelamin", options=[0, 1], format_func=lambda x: "Perempuan" if x==0 else "Laki-laki", index=1,
            help="0: Perempuan, 1: Laki-laki."
        )
        user_inputs['Age_at_enrollment'] = st.number_input(
            "Usia Saat Pendaftaran (Tahun)", min_value=17, max_value=60, value=20,
            help="Usia siswa pada saat pertama kali mendaftar."
        )
    with col3:
        user_inputs['Displaced'] = st.selectbox(
            "Pindah Domisili", options=[0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya", index=0,
            help="Apakah siswa pindah domisili untuk kuliah? (0: Tidak, 1: Ya)"
        )
        user_inputs['International'] = st.selectbox(
            "Siswa Internasional", options=[0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya", index=0,
            help="Apakah siswa berasal dari luar negeri? (0: Tidak, 1: Ya)"
        )

# --- Bagian 3: Status Akademik & Finansial ---
with st.expander("3. Status Akademik & Finansial"):
    col1, col2, col3 = st.columns(3)
    with col1:
        user_inputs['Educational_special_needs'] = st.selectbox(
            "Kebutuhan Khusus Pendidikan", options=[0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya", index=0,
            help="Apakah siswa memiliki kebutuhan khusus pendidikan? (0: Tidak, 1: Ya)"
        )
        user_inputs['Debtor'] = st.selectbox(
            "Memiliki Tunggakan", options=[0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya", index=0,
            help="Apakah siswa memiliki tunggakan pembayaran? (0: Tidak, 1: Ya)"
        )
        user_inputs['Tuition_fees_up_to_date'] = st.selectbox(
            "Pembayaran Uang Kuliah Tepat Waktu", options=[0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya", index=1,
            help="Apakah siswa membayar uang kuliah tepat waktu? (0: Tidak, 1: Ya)"
        )
    with col2:
        user_inputs['Scholarship_holder'] = st.selectbox(
            "Penerima Beasiswa", options=[0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya", index=0,
            help="Apakah siswa adalah penerima beasiswa? (0: Tidak, 1: Ya)"
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
        # Create a new DataFrame for prediction, copying the template structure and filling with defaults
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
        if 'Age_Squared' in input_df.columns:
            input_df['Age_Squared'] = input_df['Age_at_enrollment'] ** 2
        else:
            input_df['Age_Squared'] = input_df['Age_at_enrollment'] ** 2

        if 'Academic_Performance_1st_Sem' in input_df.columns:
            if input_df['Curricular_units_1st_sem_enrolled'].iloc[0] > 0:
                input_df['Academic_Performance_1st_Sem'] = (input_df['Curricular_units_1st_sem_grade'] * input_df['Curricular_units_1st_sem_approved']) / input_df['Curricular_units_1st_sem_enrolled']
            else:
                input_df['Academic_Performance_1st_Sem'] = 0.0
        else:
            if input_df['Curricular_units_1st_sem_enrolled'].iloc[0] > 0:
                input_df['Academic_Performance_1st_Sem'] = (input_df['Curricular_units_1st_sem_grade'] * input_df['Curricular_units_1st_sem_approved']) / input_df['Curricular_units_1st_sem_enrolled']
            else:
                input_df['Academic_Performance_1st_Sem'] = 0.0

        failed_courses_1st_sem = input_df['Curricular_units_1st_sem_enrolled'] - input_df['Curricular_units_1st_sem_approved']
        failed_courses_2nd_sem = input_df['Curricular_units_2nd_sem_enrolled'] - input_df['Curricular_units_2nd_sem_approved']
        total_failed_courses = failed_courses_1st_sem + failed_courses_2nd_sem

        if 'Interaction_Grade_Failed_Courses' in input_df.columns:
            input_df['Interaction_Grade_Failed_Courses'] = input_df['Admission_grade'] * total_failed_courses
        else:
            input_df['Interaction_Grade_Failed_Courses'] = input_df['Admission_grade'] * total_failed_courses

        if 'Interaction_Mother_Father_Qual' in input_df.columns:
            input_df['Interaction_Mother_Father_Qual'] = input_df['Mothers_qualification'] * input_df['Fathers_qualification']
        else:
            input_df['Interaction_Mother_Father_Qual'] = input_df['Mothers_qualification'] * input_df['Fathers_qualification']

        if 'Combined_Parents_Qual' in input_df.columns:
            input_df['Combined_Parents_Qual'] = input_df['Mothers_qualification'] + input_df['Fathers_qualification']
        else:
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

