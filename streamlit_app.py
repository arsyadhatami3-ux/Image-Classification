import os
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import urllib.request

# ======================================================
# CONFIG STREAMLIT
# ======================================================

st.set_page_config(
    page_title="Image Classification",
    layout="centered"
)

st.title("Image Classification App")
st.write("Upload gambar untuk melakukan prediksi klasifikasi.")

# ======================================================
# CONFIGURATION GOOGLE DRIVE
# ======================================================
# Nama file model target di server lokal Streamlit
model_filename = 'load_image_classification_model.h5'

# ⬇️ PASTE LINK SHARING GOOGLE DRIVE KAMU DI SINI (Ganti teks di bawah)
GDRIVE_SHARE_LINK = "https://drive.google.com/file/d/1U5coUaAXjAWRXbxtuMGzkLpz8NUeUkgQ/view?usp=sharing"


def get_direct_download_url(share_url):
    """
    Mengonversi link sharing Google Drive standar menjadi link download langsung.
    """
    if "drive.google.com" in share_url:
        if "/file/d/" in share_url:
            file_id = share_url.split("/file/d/")[1].split("/")[0]
            return f"https://docs.google.com/uc?export=download&id={file_id}"
    return share_url


# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():
    # Jika file model belum ada di direktori lokal Streamlit, unduh dari Google Drive
    if not os.path.exists(model_filename):
        if GDRIVE_SHARE_LINK == "MASUKKAN_LINK_SHARING_GOOGLE_DRIVE_DISINI" or GDRIVE_SHARE_LINK == "":
            st.error("⚠️ Tautan Google Drive belum dimasukkan di dalam source code.")
            st.stop()
            
        with st.spinner("⏳ Sedang mengunduh file model .h5 dari Google Drive (Proses ini hanya berjalan sekali)..."):
            try:
                direct_url = get_direct_download_url(GDRIVE_SHARE_LINK)
                urllib.request.urlretrieve(direct_url, model_filename)
                st.success("✅ File model berhasil diunduh dari Google Drive!")
            except Exception as e:
                st.error(f"❌ Gagal mengunduh file dari Google Drive: {e}")
                st.stop()
                
    # Memuat model TensorFlow/Keras
    model = tf.keras.models.load_model(model_filename)
    return model

# Menjalankan fungsi load model
loaded_model = load_model()

# ======================================================
# PARAMETER
# ======================================================

target_size = (128, 128)

class_labels = {
    0: 'negative',
    1: 'positive'
}

# ======================================================
# FUNGSI PREDIKSI
# ======================================================

def predict_image(uploaded_file, model):

    # Buka gambar
    img = Image.open(uploaded_file).convert('RGB')

    # Resize gambar
    img_resized = img.resize(target_size)

    # Konversi ke array
    img_array = image.img_to_array(img_resized)

    # Tambahkan batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalisasi
    img_array = img_array / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    probability = prediction[0][0]

    # Tentukan kelas
    if probability > 0.5:
        predicted_class_index = 1
    else:
        predicted_class_index = 0

    predicted_label = class_labels[predicted_class_index]

    return img, predicted_label, probability

# ======================================================
# UPLOAD GAMBAR
# ======================================================

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    # Prediksi
    img, label, probability = predict_image(
        uploaded_file,
        loaded_model
    )

    # Tampilkan gambar
    st.image(img, caption="Gambar Uploaded", use_container_width=True)

    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    st.write(f"Prediksi: {label}")
    st.write(f"Probabilitas: {probability * 100:.2f}%")
