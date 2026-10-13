from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn


LABELS = Path(
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
    nn.Linear(256, len(LABELS)),
)



weights_file = hf_hub_download("nateraw/quickdraw", "pytorch_model.bin")
state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.eval()


def prepare_image(editor_value):
    """Convert the drawing from the browser into the 28x28 input the model expects."""
    if editor_value is None:
        return None

   
    if isinstance(editor_value, dict):
        image = editor_value.get("composite")
    else:
        image = editor_value

    if image is None:
        return None

    image = Image.fromarray(np.asarray(image)).convert("L")
    image = image.resize((28, 28))

    pixels = np.asarray(image, dtype=np.float32)

    
    if pixels.mean() > 127:
        pixels = 255 - pixels

    return pixels


def predict(drawing, intended_label):
    pixels = prepare_image(drawing)

    if pixels is None:
        return {}, "Draw something first, then click **Predict**."

    x = torch.tensor(pixels, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output[0], dim=0)

    values, indices = torch.topk(probabilities, 5)

    predictions = {
        LABELS[index.item()]: score.item()
        for index, score in zip(indices, values)
    }

    top_prediction = next(iter(predictions))
    top_score = predictions[top_prediction]

    if not intended_label:
        explanation = (
            f"### AI prediction: **{top_prediction}**\n\n"
            f"Top prediction score: **{top_score:.1%}**\n\n"
            "Choose what you intended to draw if you want to compare the AI's answer with yours."
        )
    elif top_prediction == intended_label:
        explanation = (
            f"### ✅ Prediction matched\n\n"
            f"You intended: **{intended_label}**  \n"
            f"AI predicted: **{top_prediction}**  \n"
            f"Top prediction score: **{top_score:.1%}**\n\n"
            "The model gave your intended category the highest score. "
            "That does not mean the model is guaranteed to be correct on every similar drawing."
        )
    else:
        explanation = (
            f"### ❌ Prediction did not match\n\n"
            f"You intended: **{intended_label}**  \n"
            f"AI predicted: **{top_prediction}**  \n"
            f"Top prediction score: **{top_score:.1%}**\n\n"
            "The model's learned patterns led it to a different answer. "
            "A high prediction score does **not** automatically mean the prediction is correct."
        )

    return predictions, explanation


with gr.Blocks(title="AI Prediction Lab") as demo:
    gr.Markdown(
        "# 🧪 AI Prediction Lab\n"
        "Draw an object, choose what you intended to draw, and inspect the model's predictions."
    )

    with gr.Row():
        drawing = gr.Sketchpad(
            label="Draw here",
            type="numpy",
            image_mode="RGBA",
            canvas_size=(420, 420),
            fixed_canvas=True,
        )

        with gr.Column():
            intended = gr.Dropdown(
                choices=LABELS,
                label="What were you trying to draw?",
                value=None,
                filterable=True,
            )

            predict_button = gr.Button("Predict", variant="primary")

            scores = gr.Label(
                label="Top 5 model predictions",
                num_top_classes=5,
            )

            result = gr.Markdown()

    predict_button.click(
        fn=predict,
        inputs=[drawing, intended],
        outputs=[scores, result],
    )


demo.launch()