import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Flower Classifier", page_icon="🌼", layout="centered")

# ==== KONFIGURASI DASAR ====
# Ganti sesuai lokasi file kamu di Drive/PC
MODEL_PATH = "best_finetune.keras"   # contoh: "/content/drive/MyDrive/flowers_runs/best_finetune.keras"
CLASS_JSON = "class_names.json"      # opsional; jika ada akan dipakai
DEFAULT_CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# ==== UTIL =====
@st.cache_resource(show_spinner=True)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def load_class_names():
    try:
        with open(CLASS_JSON, "r") as f:
            names = json.load(f)
            if isinstance(names, dict):  # kalau disimpan sebagai dict index->name
                # pastikan urut dari 0..N-1
                keys = sorted([int(k) for k in names.keys()])
                names = [names[str(k)] for k in keys]
            return names
    except Exception:
        return DEFAULT_CLASS_NAMES

def preprocess_image(img: Image.Image, target_size):
    # img: PIL.Image
    img = img.convert("RGB").resize(target_size, Image.BILINEAR)
    x = np.array(img, dtype=np.float32)

    # Jika model sudah punya layer preprocessing/normalization di dalamnya,
    # input 0..255 boleh; tapi aman kita skala 0..1
    x = x / 255.0
    x = np.expand_dims(x, 0)  # (1, H, W, 3)
    return x

def get_input_size(model):
    # Ambil ukuran input dari model: (None, H, W, 3)
    ishape = model.inputs[0].shape
    # TensorShape(None, H, W, 3)
    H = int(ishape[1]) if ishape[1] is not None else 224
    W = int(ishape[2]) if ishape[2] is not None else 224
    return (W, H)  # PIL resize expects (W,H)

def softmax_topk(probs, class_names, k=5):
    probs = probs.flatten()
    idxs = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idxs]

# ==== UI ====
st.title("🌼 Flower Classifier")
st.caption("Upload gambar bunga lalu dapatkan prediksi kelas + probabilitas.")

with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Path model (.keras)", MODEL_PATH)
    use_tta = st.checkbox("Gunakan TTA (flip horizontal)", value=False)
    run_btn = st.button("Muat / Reload Model")

# cache model
if "model" not in st.session_state or run_btn or st.session_state.get("model_path") != model_path:
    with st.spinner("Memuat model..."):
        st.session_state["model"] = load_model(model_path)
        st.session_state["model_path"] = model_path

model = st.session_state["model"]
class_names = load_class_names()

col1, col2 = st.columns([1,1])
with col1:
    uploaded = st.file_uploader("Pilih gambar (jpg/png)", type=["jpg","jpeg","png"])
with col2:
    st.write("Kelas tersedia:")
    st.write(", ".join(class_names))

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Gambar diupload", use_container_width=True)

    # siapkan input sesuai ukuran model
    target_size = get_input_size(model)  # (W,H)
    x = preprocess_image(img, target_size)

    # Opsional: TTA sederhana (flip horizontal)
    if use_tta:
        x_flip = np.flip(x, axis=2)  # flip width
        preds1 = model.predict(x, verbose=0)
        preds2 = model.predict(x_flip, verbose=0)
        preds = (preds1 + preds2) / 2.0
    else:
        preds = model.predict(x, verbose=0)

    top5 = softmax_topk(preds[0], class_names, k=min(5, len(class_names)))

    st.subheader("Hasil Prediksi")
    st.metric("Prediksi Utama", top5[0][0], help=f"Prob: {top5[0][1]:.3f}")
    st.write("Top-K:")
    for name, p in top5:
        st.write(f"- **{name}**: {p:.3f}")

    # Simple bar chart
    st.bar_chart({name: p for name, p in top5})
else:
    st.info("Upload gambar untuk memulai.")

st.caption("Tips: jika hasil tidak sesuai, pastikan class_names.json konsisten dengan mapping saat training.")
