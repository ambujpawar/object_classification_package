"""
Run the Prediction API
"""

from fastapi import FastAPI
import urllib.request
from PIL import Image
import torch
from loguru import logger
from pydantic import BaseModel
# In module imports
from model import load_model
from transforms import get_transforms

model = load_model("resnet50", num_classes=37)
model.load_state_dict(torch.load("models/best_model_resnet50.pt", map_location=torch.device('cpu')))
model.eval()

app = FastAPI()
import json
with open('class_mapping.json', 'r') as f:
    class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}


class ImageUrl(BaseModel):
    url: str


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
def predict(image_url: ImageUrl):
    # Read image from URL using Pillow
    image = Image.open(urllib.request.urlopen(image_url.url))

    test_transforms = get_transforms()["test"]
    image = test_transforms(image)
    output = model(image.unsqueeze(0))
    _, preds = torch.max(output, 1)
    logger.info(f"Predicted class: {preds.item()}")
    pred_class = idx_to_class[preds.item()]
    logger.info(f"Predicted class: {pred_class}")
    return {"Predicted": pred_class}


@app.post("/predict_top_3/")
def predict_top_n(image_url: ImageUrl):
    # Read image from URL using Pillow
    image = Image.open(urllib.request.urlopen(image_url.url))

    test_transforms = get_transforms()["test"]
    image = test_transforms(image)
    output = model(image.unsqueeze(0))
    _, preds = torch.topk(output, 3)
    preds = preds.squeeze(0).tolist()
    logger.info(f"Predicted classes: {preds}")
    pred_classes = [idx_to_class[pred] for pred in preds]
    logger.info(f"Predicted classes: {pred_classes}")
    return {"Predicted": pred_classes}
