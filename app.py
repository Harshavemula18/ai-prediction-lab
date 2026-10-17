
from pathlib import Path

import streamlit as st
import streamlit_drawable_canvas
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn

st.set_page_config(
    page_title="AI Prediction Lab",
    page_icon="🧪",
    layout="centered",
)

st.title("🧪 AI Prediction Lab")
st.write(
    "Draw anything on the canvas, choose what you intended to draw, "
    "and see what the AI predicts!"
)


@st.cache_resource
def load_model():
    labels_path = hf_hub_download(
        "nateraw/quickdraw", "class_names.txt"
    )
    labels = Path(labels_path).read_text().splitlines()

    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1152, 256),
        nn.ReLU(),
        nn.Linear(256, len(labels)),
    )

    weights_path = hf_hub_download(
        "nateraw/quickdraw", "pytorch_model.bin"
    )
    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, labels


try:
    with st.spinner("Loading AI model..."):
        model, labels = load_model()
except Exception as error:
    st.error("Could not load the AI model.")
    st.code(str(error))
    st.stop()


st.subheader("🎨 Draw here")

st.caption("Use your mouse or trackpad to draw in the white box.")

canvas = streamlit_drawable_canvas.st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=8,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=350,
    width=420,
    drawing_mode="freedraw",
    key="ai_prediction_canvas",
)

intended_label = st.selectbox(
    "What were you trying to draw?",
    ["Choose a category..."] + labels,
)

if st.button("🔍 Predict", type="primary", use_container_width=True):
    if canvas.image_data is None:
        st.warning("Draw something first!")
        st.stop()

    rgba = canvas.image_data.astype(np.uint8)

    if not np.any(rgba[:, :, 3] > 0):
        st.warning("Draw something first!")
        st.stop()

    # Convert the drawing to grayscale.
    image = Image.fromarray(rgba).convert("RGBA")
    white_background = Image.new("RGBA", image.size, "WHITE")
    image = Image.alpha_composite(white_background, image).convert("L")

    # QuickDraw models generally expect dark strokes on a light background.
    # Invert so the drawing becomes bright on a dark background.
    pixels = np.asarray(image, dtype=np.uint8)
    pixels = 255 - pixels

    # Resize to the model's input size.
    image = Image.fromarray(pixels).resize(
        (28, 28), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(image, dtype=np.float32)

    x = torch.tensor(pixels).unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output[0], dim=0)

    values, indices = torch.topk(probabilities, 5)

    predictions = {
        labels[index.item()]: float(score.item())
        for index, score in zip(indices, values)
    }

    top_prediction = next(iter(predictions))
    top_score = predictions[top_prediction]

    st.subheader("📊 AI Predictions")
    st.bar_chart(predictions)

    st.metric("Top prediction", top_prediction)
    st.write(f"Prediction score: **{top_score:.1%}**")

    if intended_label != "Choose a category...":
        if top_prediction == intended_label:
            st.success("The AI guessed your intended drawing!")
        else:
            st.warning(
                f"You intended **{intended_label}**, "
                f"but the AI predicted **{top_prediction}**."
            )

    st.caption(
        "Prediction scores are model outputs, not guarantees of correctness."
    )

st.divider()
st.caption("Built with Python, Streamlit and PyTorch.")
