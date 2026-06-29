# Define Classification Model(s)

import torch.nn as nn
import torchvision.models as models
from torchvision.models import *


def get_model(num_classes=29, model_name="resnet18", pretrained=True):
    
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

    elif model_name == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)

    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)

    elif model_name == "resnet101":
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)

    elif model_name == "resnet152":
        weights = ResNet152_Weights.DEFAULT if pretrained else None
        model = models.resnet152(weights=weights)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model