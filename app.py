
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
    layout="centered",
)

# ---------- DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: #f8f9fb;
    color: #202020;
    font-family: Arial, Helvetica, sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1 {
    color: #202020 !important;
    font-family: Arial, Helvetica, sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #202020 !important;
    font-weight: 700 !important;
}

p, label, [data-testid="stCaptionContainer"] {
    color: #333333 !important;
    font-family: Arial, Helvetica, sans-serif !important;
}

div.stButton > button {
    background: #151515;
    color: #ffffff;
    border: 1px solid #151515;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.65rem 1rem;
}

div.stButton > button:hover {
    background: #333333;
    color: #ffffff;
    border-color: #333333;
}

[data-baseweb="select"] > div {
    background-color: #ffffff;
    border-color: #d8d8d8;
    border-radius: 8px;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 16px;
}

hr {
    border-color: #e5e5e5;
}

footer {
    background: #111111;
}
</style>
""", unsafe_allow_html=True)


# ---------- PAGE TITLE ----------
st.title("🧪 AI Prediction Lab")
st.write(
    "Draw an object, choose what you intended to draw, "
    "and discover what the AI predicts."
)


# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    labels_path = hf_hub_download(
        repo_id="nateraw/quickdraw",
        filename="class_names.txt",
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
        repo_id="nateraw/quickdraw",
        filename="pytorch_model.bin",
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


# ---------- DRAWING CANVAS ----------
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
    return_image_data=True,
)


# ---------- USER'S INTENDED DRAWING ----------
intended_label = st.selectbox(
    "What were you trying to draw?",
    ["Choose a category..."] + labels,
)


# ---------- PREDICTION ----------
if st.button("🔍 Predict", type="primary", use_container_width=True):
    try:
        if canvas is None or canvas.image_data is None:
            st.warning("Draw something first!")
            st.stop()

        rgba = canvas.image_data.astype(np.uint8)

        if not np.any(rgba[:, :, 3] > 0):
            st.warning("Draw something first!")
            st.stop()

        # Put the drawing on a white background.
        image = Image.fromarray(rgba, mode="RGBA")
        white_background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(
            white_background, image
        ).convert("L")

        # Convert dark strokes to bright strokes.
        pixels = 255 - np.asarray(image, dtype=np.uint8)

        # Resize to the model's expected 28x28 input.
        image = Image.fromarray(pixels).resize(
            (28, 28), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(image, dtype=np.float32)

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

        top_prediction = next(iter(predictions))
        top_score = predictions[top_prediction]

        st.subheader("📊 Top 5 AI predictions")
        st.bar_chart(predictions)

        st.metric("AI's top guess", top_prediction)
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

    except Exception as error:
        st.error("Something went wrong while making the prediction.")
        st.code(str(error))


# ---------- FOOTER ----------
st.divider()
st.caption("Built with Python, Streamlit and PyTorch.")
