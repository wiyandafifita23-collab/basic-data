import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- 1. Muat model dan scaler yang telah disimpan ---
# Menggunakan st.cache_resource agar model dan scaler hanya dimuat sekali
@st.cache_resource
def load_artifacts():
    try:
        with open('gradient_boosting_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        st.error("File model atau scaler tidak ditemukan. Pastikan 'gradient_boosting_model.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
        st.stop() # Hentikan aplikasi jika file tidak ditemukan

loaded_model, loaded_scaler = load_artifacts()

# st.write("Model dan Scaler berhasil dimuat.") # Bisa dinonaktifkan untuk UI yang lebih bersih

# --- 2. Rekonstruksi objek encoder dan mapping yang digunakan saat preprocessing ---

# Mapping Jenis Kelamin (dari notebook: cell QbiI5O60rA-o)
mapping_gender = {'Pria': 'Laki-laki', 'L': 'Laki-laki', 'Perempuan': 'Wanita', 'P': 'Wanita'}

# Kategori unik untuk LabelEncoder (penting agar konsisten dengan saat training)
# Berdasarkan df_bersih.unique() dari notebook dan cara LabelEncoder mengurutkan secara alfabetis:
pendidikan_categories = ['D3', 'S1', 'SMA', 'SMK'] # Urutan alfabetis dari unique values saat training
jurusan_categories = ['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'] # Urutan alfabetis

le_pendidikan = LabelEncoder()
le_pendidikan.fit(pendidikan_categories)

le_jurusan = LabelEncoder()
le_jurusan.fit(jurusan_categories)

# Nama kolom fitur saat pelatihan (dari X.columns di notebook)
training_feature_columns = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                           'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                           'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']


# --- 3. Definisikan fungsi preprocessing untuk data input baru ---
def preprocess_new_data(new_raw_data):
    df_new = pd.DataFrame([new_raw_data])

    # Terapkan mapping pada kolom 'Jenis_Kelamin'
    df_new['Jenis_Kelamin'] = df_new['Jenis_Kelamin'].replace(mapping_gender)

    # Terapkan Label Encoding
    # Gunakan .transform() dan pastikan nilai ada di classes_ LabelEncoder
    df_new['Pendidikan_encoded'] = df_new['Pendidikan'].apply(lambda x: le_pendidikan.transform([x])[0] if x in le_pendidikan.classes_ else -1) # -1 sebagai indikator error
    df_new['Jurusan_encoded'] = df_new['Jurusan'].apply(lambda x: le_jurusan.transform([x])[0] if x in le_jurusan.classes_ else -1)

    # Identifikasi kolom one-hot dari training_feature_columns
    one_hot_training_cols = [col for col in training_feature_columns if col.startswith('Jenis_Kelamin_') or col.startswith('Status_Bekerja_')]

    # Terapkan One-Hot Encoding pada 'Jenis_Kelamin' dan 'Status_Bekerja'
    df_onehot_new = pd.get_dummies(df_new[['Jenis_Kelamin', 'Status_Bekerja']], prefix=['Jenis_Kelamin', 'Status_Bekerja'])
    df_onehot_new = df_onehot_new.astype(int)

    # Pastikan semua kolom one-hot dari pelatihan ada, dan isi dengan 0 jika tidak ada
    df_onehot_final = pd.DataFrame(0, index=df_new.index, columns=one_hot_training_cols)
    for col in df_onehot_new.columns:
        if col in df_onehot_final.columns:
            df_onehot_final[col] = df_onehot_new[col]

    # Gabungkan semua fitur menjadi satu DataFrame sementara
    input_for_scaling = pd.DataFrame(index=[0])
    input_for_scaling['Usia'] = df_new['Usia'].iloc[0]
    input_for_scaling['Durasi_Jam'] = df_new['Durasi_Jam'].iloc[0]
    input_for_scaling['Nilai_Ujian'] = df_new['Nilai_Ujian'].iloc[0]
    input_for_scaling['Pendidikan'] = df_new['Pendidikan_encoded'].iloc[0] # Gunakan kolom hasil encoding
    input_for_scaling['Jurusan'] = df_new['Jurusan_encoded'].iloc[0]       # Gunakan kolom hasil encoding

    # Tambahkan kolom one-hot encoded
    input_for_scaling = pd.concat([input_for_scaling, df_onehot_final], axis=1)

    # Pastikan urutan kolom sesuai dengan training_feature_columns (X.columns)
    input_for_scaling = input_for_scaling[training_feature_columns] # Reorder to match X.columns

    # Terapkan StandardScaler
    scaled_data = loaded_scaler.transform(input_for_scaling)
    final_features_df = pd.DataFrame(scaled_data, columns=training_feature_columns)

    return final_features_df

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Prediksi Gaji Pertama", layout="centered") # Konfigurasi halaman
st.title("Prediksi Gaji Pertama Setelah Pelatihan Vokasi")
st.markdown("Aplikasi ini memprediksi estimasi gaji pertama berdasarkan data pelatihan vokasi. Masukkan data peserta di sidebar dan klik 'Prediksi Gaji'.")

# Input dari pengguna di sidebar
st.sidebar.header("Input Data Peserta")
with st.sidebar.form("input_form"): # Menggunakan form untuk mengelompokkan input
    usia = st.slider("Usia (Tahun)", min_value=18, max_value=60, value=25)
    durasi_jam = st.slider("Durasi Pelatihan (Jam)", min_value=20, max_value=100, value=60)
    nilai_ujian = st.slider("Nilai Ujian", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
    pendidikan = st.selectbox("Pendidikan Terakhir", pendidikan_categories, index=2) # 'SMA' is index 2
    jurusan = st.selectbox("Jurusan Pelatihan", jurusan_categories, index=1) # 'Desain Grafis' is index 1
    jenis_kelamin = st.radio("Jenis Kelamin", ['Laki-laki', 'Wanita'], index=1) # 'Wanita' is index 1
    status_bekerja = st.radio("Status Setelah Pelatihan", ['Sudah Bekerja', 'Belum Bekerja'], index=0) # 'Sudah Bekerja' is index 0

    submit_button = st.form_submit_button("Prediksi Gaji")

# Logika prediksi setelah tombol ditekan
if submit_button:
    new_data_input = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }

    # Preprocess data baru
    processed_input = preprocess_new_data(new_data_input)

    # Lakukan prediksi
    predicted_gaji = loaded_model.predict(processed_input)

    st.subheader("\nHasil Prediksi:")
    st.success(f"Gaji Pertama yang Diprediksi: **Rp {predicted_gaji[0]*1000000:.2f}** (sekitar {predicted_gaji[0]:.2f} Juta Rupiah)")
    st.write("\n---")
    st.info("**Penting:** Pastikan file `gradient_boosting_model.pkl` dan `scaler.pkl` berada di direktori yang sama dengan file `app.py` Anda. Jika Anda menjalankan ini di Google Colab, Anda perlu mengunduh file `.pkl` ke lingkungan lokal Anda dan menjalankannya dengan Streamlit di sana.")

else:
    st.info("Isi detail peserta di sidebar dan klik 'Prediksi Gaji' untuk melihat hasil estimasi gaji pertama.")
