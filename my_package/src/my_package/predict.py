"""
Script to predict the class of an image.
"""

import argparse
import copy
# Third party imports
from PIL import Image
import torch
import torch.nn as nn
from loguru import logger
# In module imports
from transforms import get_transforms
from model import load_model


def main(args):
    """
    Where the magic happens
    """
    # Load the model weights
    model = load_model(args.backbone, num_classes=37)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_transforms = get_transforms()["test"]
    image = test_transforms(Image.open(args.image_path))
    output = model(image.unsqueeze(0))
    _, preds = torch.max(output, 1)
    logger.info(f"Predicted class: {preds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet50", help="Name of the backbone to use")
    parser.add_argument("--model_path", type=str, help="Path to the weights of the model")
    parser.add_argument("--image_path", type=str, help="Path to the image")
    args = parser.parse_args()
    main(args)
