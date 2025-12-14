import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

# ======================
# Page config
# ======================
st.set_page_config(page_title="Flower Classification", page_icon="🌸", layout="wide")

# ======================
# CSS: fix header cut + simple UI
# ======================
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { height: 0px; }
    .block-container { padding-top: 3.5rem !important; padding-bottom: 2rem; max-width: 1200px; }
    .title { font-size: 2.3rem; font-weight: 700; margin-bottom: 0.25rem; }
    .subtitle { color: #9ca3af; margin-bottom: 1.4rem; }
    .card { padding: 1.2rem 1.4rem; border-radius: 18px; background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10); }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# Files (same folder as app.py)
# ======================
MODEL_PATH = "best_finetune.keras"
CLASS_JSON = "class_names.json"

@st.cache_resource(show_spinner=True)
def load_assets():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}")
    if not os.path.exists(CLASS_JSON):
        raise FileNotFoundError(f"File class_names tidak ditemukan: {CLASS_JSON}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(CLASS_JSON, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or len(class_names) == 0:
        raise ValueError("class_names.json harus berisi list, contoh: [\"daisy\", \"roses\", ...]")

    # detect model input size
    ishape = model.inputs[0].shape  # (None, H, W, 3)
    H = int(ishape[1]) if ishape[1] is not None else 224
    W = int(ishape[2]) if ishape[2] is not None else 224

    return model, class_names, (H, W)

def preprocess_image(img: Image.Image, target_hw):
    """Match training: keep pixels 0..255 (no /255)."""
    h, w = target_hw
    img = img.convert("RGB").resize((w, h))
    x = np.array(img, dtype=np.float32)  # 0..255
    return np.expand_dims(x, axis=0)

# ======================
# Load model
# ======================
try:
    model, CLASS_NAMES, TARGET_HW = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ======================
# Header
# ======================
st.markdown('<div class="title">🌸 Flower Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Unggah gambar bunga, lalu sistem memprediksi jenisnya dan menampilkan probabilitas.</div>',
    unsafe_allow_html=True
)

# ======================
# Layout
# ======================
left, right = st.columns([1.05, 1], gap="large")

with left:
    uploaded = st.file_uploader("Upload gambar (JPG / PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Silakan upload gambar untuk memulai.")
        st.stop()

    img = Image.open(uploaded)
    st.image(img, caption="Gambar input", use_container_width=True)

with right:
    # Predict
    x = preprocess_image(img, TARGET_HW)
    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx]) * 100.0

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Hasil Prediksi")
    st.metric("Jenis Bunga", pred_label, f"{pred_conf:.2f}%")
    st.progress(min(max(pred_conf / 100.0, 0.0), 1.0))

    # Probability chart
    st.markdown("### Distribusi Probabilitas")
    df = pd.DataFrame(
        {
            "Jenis Bunga": CLASS_NAMES,
            "Probabilitas (%)": (probs * 100.0).astype(float),
        }
    ).sort_values("Probabilitas (%)", ascending=False)

    st.bar_chart(df.set_index("Jenis Bunga"), height=320)

    st.markdown("### Top-5")
    top5 = df.head(5)
    for _, r in top5.iterrows():
        st.write(f"- **{r['Jenis Bunga']}**: {r['Probabilitas (%)']:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)
