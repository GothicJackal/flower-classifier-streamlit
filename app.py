import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Flower Classification", page_icon="🌸", layout="wide")

# ---------- Simple styling ----------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      .title { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
      .subtitle { color: #6b7280; margin-bottom: 1.2rem; }
      .card {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(255,255,255,0.60);
        box-shadow: 0 6px 24px rgba(0,0,0,0.06);
      }
      .big-label { font-size: 1.4rem; font-weight: 700; margin: 0; }
      .small-muted { color: #6b7280; margin-top: 0.2rem; }
      .hr { margin: 0.9rem 0 1.1rem; border-bottom: 1px solid rgba(0,0,0,0.08); }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Paths ----------
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

    ishape = model.inputs[0].shape  # (None, H, W, 3)
    H = int(ishape[1]) if ishape[1] is not None else 224
    W = int(ishape[2]) if ishape[2] is not None else 224
    return model, class_names, (H, W)

def preprocess_image(img: Image.Image, target_hw):
    """Match training: keep pixels 0–255 (no /255)."""
    h, w = target_hw
    img = img.convert("RGB").resize((w, h))
    x = np.array(img, dtype=np.float32)  # 0..255
    return np.expand_dims(x, axis=0)

# ---------- Load ----------
try:
    model, CLASS_NAMES, TARGET_HW = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ---------- Header ----------
st.markdown('<div class="title">🌸 Flower Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload gambar bunga, lalu sistem memprediksi jenisnya dan menampilkan tingkat keyakinan.</div>', unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Pengaturan")
    topk = st.slider("Jumlah kelas ditampilkan", 3, min(10, len(CLASS_NAMES)), 5)
    show_table = st.checkbox("Tampilkan tabel detail", value=False)
    st.caption(f"Input model: {TARGET_HW[0]}×{TARGET_HW[1]}")

# ---------- Main layout ----------
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    uploaded = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Silakan upload gambar untuk memulai.")
        st.stop()

    image = Image.open(uploaded)
    st.image(image, caption="Gambar input", use_container_width=True)

with right:
    x = preprocess_image(image, TARGET_HW)
    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx]) * 100

    # Result card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<p class="big-label">Prediksi: {pred_label}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="small-muted">Confidence: {pred_conf:.2f}%</p>', unsafe_allow_html=True)
    st.progress(min(max(pred_conf / 100.0, 0.0), 1.0))
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Top-k table for chart + progress
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability (%)": [float(p) * 100 for p in probs],
    }).sort_values("Probability (%)", ascending=False)

    df_top = df.head(topk).copy()

    st.subheader("Top Probabilities")
    for _, row in df_top.iterrows():
        name = row["Class"]
        p = float(row["Probability (%)"])
        st.write(f"**{name}** — {p:.2f}%")
        st.progress(min(max(p / 100.0, 0.0), 1.0))

    st.subheader("Chart")
    st.bar_chart(df_top.set_index("Class"))

    if show_table:
        st.subheader("Detail")
        st.dataframe(df, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
