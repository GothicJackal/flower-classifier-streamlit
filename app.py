import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Flower Classifier", page_icon="🌸", layout="centered")

# === File names (must be in the same folder as app.py) ===
MODEL_PATH = "best_finetune.keras"
CLASS_JSON = "class_names.json"


@st.cache_resource(show_spinner=True)
def load_assets():
    # Validate files exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Put it next to app.py (same folder)."
        )
    if not os.path.exists(CLASS_JSON):
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_JSON}. Put it next to app.py (same folder)."
        )

    # Load class names
    with open(CLASS_JSON, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or len(class_names) == 0:
        raise ValueError("class_names.json must be a non-empty JSON list, e.g. [\"daisy\", ...]")

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Detect model input size (H, W)
    ishape = model.inputs[0].shape  # (None, H, W, 3)
    h = int(ishape[1]) if ishape[1] is not None else 224
    w = int(ishape[2]) if ishape[2] is not None else 224

    return model, class_names, (h, w)


def preprocess_pil(img: Image.Image, target_hw):
    h, w = target_hw
    img = img.convert("RGB").resize((w, h))
    x = np.array(img, dtype=np.float32)

    # If your training pipeline already used preprocess_input inside the model graph
    # (common), scaling 0..1 is fine. This is a safe default for many saved models.
    # x = x / 255.0  # turn off scaling

    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)
    return x


def topk(probs, class_names, k=5):
    probs = probs.flatten()
    k = min(k, len(class_names))
    idxs = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idxs]


# === UI ===
st.title("🌸 Flower Classifier")
st.write("Upload gambar bunga, lalu model akan memprediksi kelasnya.")

with st.expander("📌 Pastikan file ada"):
    st.code(
        "Letakkan file ini dalam folder yang sama:\n"
        "- app.py\n"
        "- best_finetune.keras\n"
        "- class_names.json\n",
        language="text",
    )

# Load model + classes
try:
    model, CLASS_NAMES, TARGET_HW = load_assets()
except Exception as e:
    st.error(str(e))
    st.stop()

st.caption(f"Model input size: {TARGET_HW[0]}x{TARGET_HW[1]} | Classes: {len(CLASS_NAMES)}")

uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Silakan upload gambar untuk mulai.")
    st.stop()

img = Image.open(uploaded)
st.image(img, caption="Uploaded image", use_container_width=True)

x = preprocess_pil(img, TARGET_HW)

# Predict
probs = model.predict(x, verbose=0)[0]
pred_idx = int(np.argmax(probs))
pred_name = CLASS_NAMES[pred_idx]
pred_prob = float(probs[pred_idx])

st.subheader("Hasil Prediksi")
st.metric("Prediksi", pred_name, f"{pred_prob*100:.2f}%")

st.write("Top predictions:")
top5 = topk(probs, CLASS_NAMES, k=5)
for name, p in top5:
    st.write(f"- **{name}**: {p*100:.2f}%")

# Bar chart
chart_data = {name: p for name, p in top5}
st.bar_chart(chart_data)

# Optional: show raw probabilities
with st.expander("Lihat semua probabilitas"):
    for i, cname in enumerate(CLASS_NAMES):
        st.write(f"{cname}: {float(probs[i])*100:.2f}%")

