
# Generate the Streamlit app code
streamlit_code = f"""
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load model and scaler
try:
    with open('gradient_boosting_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        loaded_scaler = pickle.load(file)
    st.success("Model dan Scaler berhasil dimuat.")
except FileNotFoundError:
    st.error("Error: Model atau Scaler file tidak ditemukan. Pastikan 'gradient_boosting_model.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model/scaler: {{e}}")
    st.stop()

# Hardcoded categories and feature columns from training
PENDIDIKAN_CATEGORIES = {pendidikan_categories}
JURUSAN_CATEGORIES = {jurusan_categories}
TRAINING_FEATURE_COLUMNS = {feature_cols_list}

# Reconstruct LabelEncoders
le_pendidikan = LabelEncoder()
le_pendidikan.fit(PENDIDIKAN_CATEGORIES)

le_jurusan = LabelEncoder()
le_jurusan.fit(JURUSAN_CATEGORIES)

# Mapping gender used during preprocessing
mapping_gender = {{'Pria': 'Laki-laki', 'L': 'Laki-laki', 'Perempuan': 'Wanita', 'P': 'Wanita'}}

def preprocess_input(input_data):
    df_new = pd.DataFrame([input_data])

    # Apply gender mapping
    df_new['Jenis_Kelamin'] = df_new['Jenis_Kelamin'].replace(mapping_gender)

    # Label Encoding
    df_new['Pendidikan_encoded'] = le_pendidikan.transform(df_new['Pendidikan'])
    df_new['Jurusan_encoded'] = le_jurusan.transform(df_new['Jurusan'])

    # Identify one-hot columns from training features
    one_hot_training_cols = [
        col for col in TRAINING_FEATURE_COLUMNS
        if col.startswith('Jenis_Kelamin_') or col.startswith('Status_Bekerja_')
    ]

    # Apply One-Hot Encoding
    df_onehot_new = pd.get_dummies(df_new[['Jenis_Kelamin', 'Status_Bekerja']], prefix=['Jenis_Kelamin', 'Status_Bekerja'])
    df_onehot_new = df_onehot_new.astype(int)

    # Ensure all training one-hot columns are present, fill with 0 if not
    for col in one_hot_training_cols:
        if col not in df_onehot_new.columns:
            df_onehot_new[col] = 0

    # Combine all features for scaling
    input_for_scaling = pd.DataFrame(index=[0])
    input_for_scaling['Usia'] = df_new['Usia']
    input_for_scaling['Durasi_Jam'] = df_new['Durasi_Jam']
    input_for_scaling['Nilai_Ujian'] = df_new['Nilai_Ujian']
    input_for_scaling['Pendidikan'] = df_new['Pendidikan_encoded']
    input_for_scaling['Jurusan'] = df_new['Jurusan_encoded']

    for col in one_hot_training_cols:
        input_for_scaling[col] = df_onehot_new[col]

    # Reorder columns to match the training feature columns
    input_for_scaling = input_for_scaling[TRAINING_FEATURE_COLUMNS]

    # Apply StandardScaler
    scaled_data = loaded_scaler.transform(input_for_scaling)
    final_features_df = pd.DataFrame(scaled_data, columns=TRAINING_FEATURE_COLUMNS)

    return final_features_df

st.set_page_config(layout="wide")

st.title("Prediksi Gaji Pertama Lulusan Pelatihan Vokasi")
st.markdown("Aplikasi ini memprediksi estimasi gaji pertama (dalam juta Rupiah) bagi lulusan pelatihan vokasi berdasarkan beberapa parameter.")

with st.sidebar:
    st.header("Input Data Peserta")
    st.markdown("Isi data berikut untuk mendapatkan prediksi gaji:")

    with st.form("input_form"):
        usia = st.slider("Usia", min_value=18, max_value=60, value=25)
        jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Wanita'])
        pendidikan = st.selectbox("Pendidikan", PENDIDIKAN_CATEGORIES)
        jurusan = st.selectbox("Jurusan", JURUSAN_CATEGORIES)
        durasi_jam = st.slider("Durasi Pelatihan (Jam)", min_value=20, max_value=100, value=60)
        nilai_ujian = st.slider("Nilai Ujian Akhir", min_value=50.0, max_value=100.0, value=75.0, step=0.1)
        status_bekerja = st.selectbox("Status Pekerjaan Saat Ini", ['Belum Bekerja', 'Sudah Bekerja'])

        submitted = st.form_submit_button("Prediksi Gaji")

if submitted:
    new_data = {
        'Usia': usia,
        'Jenis_Kelamin': jenis_kelamin,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Status_Bekerja': status_bekerja
    }

    try:
        processed_data = preprocess_input(new_data)
        predicted_gaji = loaded_model.predict(processed_data)[0]

        st.subheader("Hasil Prediksi Gaji Pertama")
        st.metric(label="Estimasi Gaji Pertama (Juta Rupiah)", value=f"Rp {predicted_gaji:.2f} Juta")

        st.subheader("Detail Input Data Anda:")
        st.write(pd.DataFrame([new_data]))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses atau memprediksi data: {e}")

st.markdown("""
<style>
.stMetric > div > div > div {
    font-size: 3em;
    font-weight: bold;
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)
"""

# Display the Streamlit app code
print("Berikut adalah kode untuk aplikasi Streamlit Anda:")
print("```python")
print(streamlit_code)
print("```")

# Provide instructions to run Streamlit
print("\nUntuk menjalankan aplikasi Streamlit ini:")
print("1. Salin kode di atas.")
print("2. Simpan kode tersebut ke dalam sebuah file, misalnya `app.py`, di direktori yang sama dengan file model (`gradient_boosting_model.pkl`) dan scaler (`scaler.pkl`).")
print("3. Buka terminal atau command prompt.")
print("4. Pastikan Anda telah menginstal Streamlit (`pip install streamlit pandas scikit-learn`).")
print("5. Navigasi ke direktori tempat Anda menyimpan `app.py`.")
print("6. Jalankan perintah: `streamlit run app.py`")
print("\nAplikasi akan terbuka di browser web default Anda.")
