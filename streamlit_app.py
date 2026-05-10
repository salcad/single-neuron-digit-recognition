from pathlib import Path

import copy
import json
from typing import Optional
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


def normalize_drawing(json_data: Optional[dict]) -> Optional[dict]:
    if not json_data:
        return None
    objects = json_data.get("objects", [])
    if not objects:
        return None
    return copy.deepcopy(json_data)


def serialize_drawing(json_data: Optional[dict]) -> Optional[str]:
    if json_data is None:
        return None
    return json.dumps(json_data, sort_keys=True)


def undo_canvas() -> None:
    history = st.session_state.canvas_history
    if not history:
        return
    history.pop()
    previous_drawing = copy.deepcopy(history[-1]) if history else None
    st.session_state.canvas_drawing = previous_drawing
    st.session_state.canvas_serialized = serialize_drawing(previous_drawing)
    st.session_state.canvas_version += 1


st.set_page_config(page_title="Digit Classifier", page_icon="✍️")
st.markdown("<h2>Handwritten Digit Classifier (0 vs 1)</h2>", unsafe_allow_html=True)
st.write("Draw a `0` or `1`, then review the model prediction.")
st.markdown(
    """
    <style>
    div[data-testid="stButton"] > button {
        color: white;
        border: 1px solid white;
        background: transparent;
    }
    div[data-testid="stButton"] > button:hover {
        color: white;
        border-color: white;
        background: rgba(255, 255, 255, 0.08);
    }
    div[data-testid="stButton"] > button:disabled {
        color: #808080;
        border-color: #808080;
        background: transparent;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not WEIGHTS_PATH.exists() or not BIAS_PATH.exists():
    st.error("Model files are missing. Run `python3 single_neuron_mnist.py` first.")
    st.stop()

weights, bias = load_model()

if "canvas_history" not in st.session_state:
    st.session_state.canvas_history = []
if "canvas_drawing" not in st.session_state:
    st.session_state.canvas_drawing = None
if "canvas_serialized" not in st.session_state:
    st.session_state.canvas_serialized = None
if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0

col1, col2 = st.columns([1.2, 1])

with col1:
    st.button(
        "Clear",
        on_click=undo_canvas,
        disabled=not st.session_state.canvas_history,
        use_container_width=False,
    )
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=16,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        initial_drawing=st.session_state.canvas_drawing,
        display_toolbar=False,
        key=f"canvas_{st.session_state.canvas_version}",
    )

with col2:
    st.write("Model files loaded:")
    st.code(f"{WEIGHTS_PATH.name}\n{BIAS_PATH.name}")

current_drawing = normalize_drawing(canvas_result.json_data)
current_serialized = serialize_drawing(current_drawing)

if current_serialized != st.session_state.canvas_serialized:
    st.session_state.canvas_drawing = copy.deepcopy(current_drawing)
    st.session_state.canvas_serialized = current_serialized
    if current_drawing is None:
        st.session_state.canvas_history = []
    else:
        st.session_state.canvas_history.append(copy.deepcopy(current_drawing))
    st.rerun()

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
