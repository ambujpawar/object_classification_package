"""
File aiming to support all the image classification models supported by torchvision.
"""
import torch.nn as nn
from torchvision import models

def load_model(model_backbone: str, num_classes=37):
    """
    A function to load a model architecture from torchvision.
    Add 2 layers on top of the model to get the output of size num_classes.
    Params:
        model_backbone: backbone to use
        num_classes: number of classes
    """
    if model_backbone == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_backbone == "resnet34":
        model = models.resnet34(pretrained=True)
    elif model_backbone == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Model not supported")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    # Use resnet as a backbone and a 2 layer Neural network in front of it
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                             nn.ReLU(),
                             nn.Linear(128, num_classes)
                             )
    return model
