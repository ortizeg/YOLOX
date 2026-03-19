import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


@pytest.fixture
def random_rgb_tensor():
    """Factory fixture: returns a function that creates random RGB tensors."""
    def _make(batch_size=2, height=640, width=640):
        return torch.randn(batch_size, 3, height, width)
    return _make


@pytest.fixture
def sample_targets():
    """Factory fixture: creates sample training targets.

    Each target row: [class_id, cx, cy, w, h] in absolute pixel coords.
    Padded to max_labels rows.
    """
    def _make(batch_size=2, num_gt=5, num_classes=80, max_labels=50, img_size=640):
        targets = torch.zeros(batch_size, max_labels, 5)
        for b in range(batch_size):
            for i in range(num_gt):
                cls = torch.randint(0, num_classes, (1,)).float()
                cx = torch.rand(1) * img_size
                cy = torch.rand(1) * img_size
                w = torch.rand(1) * img_size * 0.3 + 10
                h = torch.rand(1) * img_size * 0.3 + 10
                targets[b, i] = torch.tensor([cls, cx, cy, w, h])
        return targets
    return _make


@pytest.fixture
def yolox_s_model():
    """Create a yolox-s model (depth=0.33, width=0.50)."""
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(0.33, 0.50, in_channels=in_channels)
    head = YOLOXHead(80, 0.50, in_channels=in_channels)
    model = YOLOX(backbone, head)
    return model


@pytest.fixture
def yolox_s_model_initialized():
    """Create and initialize a yolox-s model like the Exp class does."""
    import torch.nn as nn
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(0.33, 0.50, in_channels=in_channels)
    head = YOLOXHead(80, 0.50, in_channels=in_channels)
    model = YOLOX(backbone, head)
    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    return model
