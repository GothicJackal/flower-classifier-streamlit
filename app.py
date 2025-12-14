import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# =======================
# FIX HEADER & LAYOUT
# =======================
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        height: 0px;
    }
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .card {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =======================
# PATHS
# =======================
MODEL_PATH = "best_finetune.keras"
CLASS_JSON = "class_names.json"

@st.cache_resource(show_spinner=True)
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_JSON, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    ishape = model.inputs[0].shape
    H = int(ishape[1]) if ishape[1] else 224
    W = int(ishape[2]) if ishape[2] else 224
    return model, class_names, (H, W)

def preprocess_image(img: Image.Image, target_hw):
    h, w = target_hw
    img = img.convert("RGB").resize((w, h))
    x = np.array(img, dtype=np.float32)   # 0–255 (match training)
    return np.expand_dims(x, axis=0)

# =======================
# LOAD MODEL
# =======================
try:
    model, CLASS_NAMES, TARGET_HW = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =======================
# HEADER
# =======================
st.markdown('<div class="title">🌸 Flower Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Unggah gambar bunga, sistem akan memprediksi jenisnya dan menampilkan tingkat keyakinan.</div>',
    unsafe_allow_html=True
)

# =======================
# MAIN LAYOUT
# =======================
col_left, col_right = st.columns([1.05, 1], gap="large")

with col_left:
    uploaded = st.file_uploader(
        "Upload gambar (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is None:
        st.info("Silakan upload gambar untuk memulai.")
        st.stop()

    image = Image.open(uploaded)
    st.image(image, caption="Gambar input", use_container_width=True)

with col_right:
    x = preprocess_image(image, TARGET_HW)
    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx]) * 100

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")
    st.metric("Jenis Bunga", pred_label, f"{pred_conf:.2f}%")
    st.progress(pred_conf / 100)

    st.markdown("### Distribusi Probabilitas")
    df = pd.DataFrame({
        "Jenis Bunga": CLASS_NAMES,
        "Probabilitas (%)": [float]()
