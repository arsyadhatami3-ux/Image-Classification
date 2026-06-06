import os
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

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
# LOAD MODEL
# ======================================================

model_filename = 'load_image_classification_model.h5'

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_filename)
    return model

if os.path.exists(model_filename):
    loaded_model = load_model()
    st.success(f"Model '{model_filename}' berhasil dimuat.")
else:
    st.error(f"Model '{model_filename}' tidak ditemukan.")
    st.stop()

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
