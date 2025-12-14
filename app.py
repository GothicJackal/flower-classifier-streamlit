import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# === File paths (same folder as app.py) ===
MODEL_PATH = "best_finetune.keras"
CLASS_JSON = "class_names.json"

@st.cache_resource(show_spinner=True)
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(CLASS_JSON, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # detect input size from model
    ishape = model.inputs[0].shape  # (None, H, W, 3)
    H = int(ishape[1]) if ishape[1] is not None else 224
    W = int(ishape[2]) if ishape[2] is not None else 224

    return model, class_names, (H, W)

def preprocess_image(img: Image.Image, target_hw):
    """Match training preprocessing: pixel range 0–255"""
    h, w = target_hw
    img = img.convert("RGB").resize((w, h))
    x = np.array(img, dtype=np.float32)   # 0..255
    x = np.expand_dims(x, axis=0)         # (1, H, W, 3)
    return x

# === Load model & classes ===
try:
    model, CLASS_NAMES, TARGET_HW = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# === UI ===
st.title("🌸 Flower Classification")
st.write("Unggah gambar bunga, lalu sistem akan memprediksi jenis bunganya.")

uploaded = st.file_uploader(
    "Upload gambar bunga",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Silakan unggah gambar untuk memulai.")
    st.stop()

image = Image.open(uploaded)
st.image(image, caption="Gambar input", use_container_width=True)

# === Prediction ===
x = preprocess_image(image, TARGET_HW)
probs = model.predict(x, verbose=0)[0]

pred_idx = int(np.argmax(probs))
pred_label = CLASS_NAMES[pred_idx]
pred_conf = float(probs[pred_idx]) * 100

st.subheader("Hasil Prediksi")
st.metric("Jenis Bunga", pred_label, f"{pred_conf:.2f}%")

st.write("Probabilitas kelas:")
for cname, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True):
    st.write(f"- {cname}: {float(p)*100:.2f}%")
