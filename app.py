
from pathlib import Path

import streamlit as st
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn

st.set_page_config(page_title="AI Prediction Lab", page_icon="🧪")

st.title("🧪 AI Prediction Lab")
st.write("Draw an object, choose what you intended to draw, and inspect the AI's predictions.")

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

    weights_file = hf_hub_download(
        "nateraw/quickdraw", "pytorch_model.bin"
    )
    state_dict = torch.load(
        weights_file, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, labels


try:
    with st.spinner("Loading the AI model..."):
        model, labels = load_model()
except Exception as error:
    st.error("The AI model could not be loaded.")
    st.code(str(error))
    st.stop()

st.info("Draw your object in a drawing app, save it as a PNG image, and upload it below.")

uploaded_file = st.file_uploader(
    "Upload your drawing (PNG or JPG)",
    type=["png", "jpg", "jpeg"],
)

intended_label = st.selectbox(
    "What were you trying to draw?",
    ["Choose a category..."] + labels,
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Your drawing", width=300)

    if st.button("Predict", type="primary"):
        image = image.resize((28, 28))
        pixels = np.asarray(image, dtype=np.float32)

        if pixels.mean() > 127:
            pixels = 255 - pixels

        x = torch.tensor(pixels).unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = model(x)
            probabilities = torch.softmax(output[0], dim=0)

        values, indices = torch.topk(probabilities, 5)

        predictions = {
            labels[index.item()]: score.item()
            for index, score in zip(indices, values)
        }

        top_prediction = next(iter(predictions))
        top_score = predictions[top_prediction]

        st.subheader("Top 5 predictions")
        st.bar_chart(predictions)

        st.write(f"**AI prediction:** {top_prediction}")
        st.write(f"**Prediction score:** {top_score:.1%}")

        if intended_label != "Choose a category...":
            if top_prediction == intended_label:
                st.success(f"The prediction matched your intended drawing: {intended_label}")
            else:
                st.warning(
                    f"You intended to draw {intended_label}, "
                    f"but the AI predicted {top_prediction}."
                )
