
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

# Custom background and styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #172554 50%,
        #312e81 100%
    );
}

[data-testid="stHeader"] {
    background: transparent;
}

h1 {
    color: #93c5fd !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #bfdbfe !important;
}

p, label, .stMarkdown {
    color: #e2e8f0;
}

div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    padding: 0.65rem 1rem;
}

div.stButton > button:hover {
    border: 1px solid #93c5fd;
    color: white;
}

hr {
    border-color: #475569;
}

[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧪 AI Prediction Lab")
st.write(
    "Draw an object, choose what you intended to draw, "
    "and discover what the AI predicts."
)


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

intended_label = st.selectbox(
    "What were you trying to draw?",
    ["Choose a category..."] + labels,
)

if st.button("🔍 Predict", type="primary", use_container_width=True):
    try:
        if canvas is None or canvas.image_data is None:
            st.warning("Draw something first!")
            st.stop()

        rgba = canvas.image_data.astype(np.uint8)

        if not np.any(rgba[:, :, 3] > 0):
            st.warning("Draw something first!")
            st.stop()

        # Place the drawing on a white background.
        image = Image.fromarray(rgba, mode="RGBA")
        white_background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(
            white_background, image
        ).convert("L")

        # Prepare the image for the model.
        pixels = 255 - np.asarray(image, dtype=np.uint8)
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

st.divider()
st.caption("Built with Python, Streamlit and PyTorch.")
