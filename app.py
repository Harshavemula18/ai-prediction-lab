
from pathlib import Path

import streamlit as st
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="AI Prediction Lab",
    page_icon="🧪",
)

st.title("🧪 AI Prediction Lab")
st.write("Draw directly below and let the AI guess your drawing!")


@st.cache_resource
def load_model():
    labels = Path(
        hf_hub_download("nateraw/quickdraw", "class_names.txt")
    ).read_text().splitlines()

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

    weights = hf_hub_download(
        "nateraw/quickdraw", "pytorch_model.bin"
    )
    state = torch.load(
        weights, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=False)
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
st.caption("Use your mouse or trackpad inside the white canvas.")

canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=8,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=350,
    width=420,
    drawing_mode="freedraw",
    key="ai_prediction_canvas",
    update_streamlit=True,
)

intended = st.selectbox(
    "What were you trying to draw?",
    ["Choose a category..."] + labels,
)

if st.button("🔍 Predict", type="primary", use_container_width=True):
    try:
        # Read the canvas data safely.
        if canvas is None:
            st.warning("Draw something first!")
            st.stop()

        rgba = canvas.image_data

        if rgba is None or not np.any(rgba[:, :, 3] > 0):
            st.warning("Draw something first!")
            st.stop()

        # Convert the canvas to grayscale.
        rgba = rgba.astype(np.uint8)
        picture = Image.fromarray(rgba, mode="RGBA")
        white = Image.new("RGBA", picture.size, "white")
        picture = Image.alpha_composite(white, picture).convert("L")

        # Convert dark strokes to bright strokes for the model.
        pixels = 255 - np.asarray(picture, dtype=np.uint8)

        picture = Image.fromarray(pixels).resize(
            (28, 28), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(picture, dtype=np.float32)

        tensor = (
            torch.tensor(pixels)
            .unsqueeze(0)
            .unsqueeze(0)
            / 255.0
        )

        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output[0], dim=0)

        values, indices = torch.topk(probabilities, 5)

        predictions = {
            labels[index.item()]: float(score.item())
            for index, score in zip(indices, values)
        }

        guess = next(iter(predictions))

        st.subheader("📊 Top 5 predictions")
        st.bar_chart(predictions)
        st.metric("AI's guess", guess)
        st.write(f"Prediction score: **{predictions[guess]:.1%}**")

        if intended != "Choose a category...":
            if guess == intended:
                st.success("The AI guessed correctly!")
            else:
                st.warning(
                    f"You intended **{intended}**, "
                    f"but the AI guessed **{guess}**."
                )

    except Exception as error:
        st.error("Something went wrong while reading the drawing.")
        st.code(str(error))

st.divider()
st.caption("Built with Python, Streamlit and PyTorch.")
