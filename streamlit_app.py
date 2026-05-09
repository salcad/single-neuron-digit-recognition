from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
WEIGHTS_PATH = ARTIFACTS_DIR / "neuron_weights.npy"
BIAS_PATH = ARTIFACTS_DIR / "neuron_bias.npy"
RESAMPLING_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


@st.cache_resource
def load_model() -> tuple[np.ndarray, float]:
    weights = np.load(WEIGHTS_PATH)
    bias = float(np.load(BIAS_PATH))
    return weights, bias


def preprocess_image(image_data: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image_data.astype(np.uint8)).convert("L")
    image_array = np.array(image, dtype=np.float32)
    mask = image_array > 20

    if not np.any(mask):
        return np.zeros((28, 28), dtype=np.float32)

    rows, cols = np.where(mask)
    digit = image_array[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    height, width = digit.shape
    scale = min(20.0 / height, 20.0 / width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized_digit = Image.fromarray(digit.astype(np.uint8)).resize(
        (resized_width, resized_height),
        RESAMPLING_BILINEAR,
    )
    normalized_digit = np.array(resized_digit, dtype=np.float32)

    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - resized_height) // 2
    left = (28 - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = normalized_digit
    return canvas / 255.0


def predict(image_array: np.ndarray, weights: np.ndarray, bias: float) -> tuple[float, int]:
    x = image_array.flatten().reshape(1, -1)
    probability = float(sigmoid(x @ weights + bias)[0, 0])
    prediction = int(probability >= 0.5)
    return probability, prediction


st.set_page_config(page_title="Digit Classifier", page_icon="✍️")
st.title("Handwritten Digit Classifier (0 vs 1)")
st.write("Draw a `0` or `1`, then review the model prediction.")

if not WEIGHTS_PATH.exists() or not BIAS_PATH.exists():
    st.error("Model files are missing. Run `python3 single_neuron_mnist.py` first.")
    st.stop()

weights, bias = load_model()

col1, col2 = st.columns([1.2, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=16,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.write("Model files loaded:")
    st.code(f"{WEIGHTS_PATH.name}\n{BIAS_PATH.name}")

if canvas_result.image_data is None:
    st.stop()

image_array = preprocess_image(canvas_result.image_data)

if float(image_array.max()) == 0.0:
    st.info("Start drawing to see a prediction.")
    st.stop()

probability, prediction = predict(image_array, weights, bias)

preview_col, result_col = st.columns([1, 1])

with preview_col:
    st.image(image_array, caption="Preprocessed 28x28 image", width=180, clamp=True)

with result_col:
    st.metric("Probability of being 1", f"{probability:.3f}")
    st.metric("Prediction", str(prediction))
