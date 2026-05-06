"""
model.py — MobileNetV2 fine-tuned cho bài toán phân loại X-quang COVID
"""
import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES


def build_model() -> nn.Module:
    """
    Dùng MobileNetV2 pretrained ImageNet, thay classifier head
    để phân loại 3 lớp: COVID / Normal / Viral Pneumonia.

    Tại sao MobileNetV2?
      - Nhẹ (~3.4M params) → phù hợp federated learning (ít băng thông)
      - Pretrained → hội tụ nhanh dù mỗi client ít data
      - Accuracy tốt trên ảnh y tế grayscale chuyển RGB
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Đóng băng backbone, chỉ train classifier head
    for param in model.features.parameters():
        param.requires_grad = False

    # Thay classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, NUM_CLASSES),
    )

    return model


def get_parameters(model: nn.Module) -> list:
    """Lấy weights model dưới dạng list numpy arrays (gửi qua mạng FL)."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list) -> None:
    """Nạp weights nhận từ server vào model."""
    import numpy as np
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)