# Define Classification Model(s)

import torch.nn as nn
import torchvision.models as models


def get_model(num_classes=30, model_name="resnet18", pretrained=True):
    """
    Returns a classification model.

    Args:
        num_classes (int): number of output classes (4 for xBD)
        model_name (str): model architecture
        pretrained (bool): use ImageNet pretrained weights
    """

    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model