import pytest
import torch

from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead


NUM_CLASSES = 80
IN_CHANNELS = [256, 512, 1024]
INPUT_SIZE = 320
BATCH_SIZE = 1
TRAINING_LOSS_KEYS = {"total_loss", "iou_loss", "l1_loss", "conf_loss", "cls_loss", "num_fg"}


def _build_model(depth, width, depthwise=False, num_classes=NUM_CLASSES):
    backbone = YOLOPAFPN(depth, width, in_channels=IN_CHANNELS, depthwise=depthwise, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=IN_CHANNELS, depthwise=depthwise, act="silu")
    return YOLOX(backbone, head)


def _dummy_input(batch_size=BATCH_SIZE, size=INPUT_SIZE):
    return torch.randn(batch_size, 3, size, size)


def _dummy_targets(batch_size=BATCH_SIZE, max_labels=5):
    targets = torch.zeros(batch_size, max_labels, 5)
    # One valid target per image: class 0, centered, small box
    targets[:, 0, 0] = 0  # class_id
    targets[:, 0, 1] = 0.5  # cx
    targets[:, 0, 2] = 0.5  # cy
    targets[:, 0, 3] = 0.1  # w
    targets[:, 0, 4] = 0.1  # h
    return targets


@pytest.fixture
def small_model():
    return _build_model(0.33, 0.50)


class TestYOLOXTraining:
    """Tests for YOLOX in training mode."""

    def test_forward_returns_dict_with_expected_keys(self, small_model):
        small_model.train()
        x = _dummy_input()
        targets = _dummy_targets()
        output = small_model(x, targets=targets)
        assert isinstance(output, dict)
        assert set(output.keys()) == TRAINING_LOSS_KEYS

    def test_total_loss_has_grad_fn_and_all_losses_finite(self, small_model):
        small_model.train()
        x = _dummy_input()
        targets = _dummy_targets()
        output = small_model(x, targets=targets)
        assert output["total_loss"].grad_fn is not None, "total_loss should have grad_fn"
        for key in TRAINING_LOSS_KEYS:
            val = output[key]
            if isinstance(val, torch.Tensor) and val.numel() == 1:
                assert torch.isfinite(val).item(), f"{key} is not finite: {val.item()}"

    def test_gradient_flow_to_backbone(self, small_model):
        small_model.train()
        x = _dummy_input()
        targets = _dummy_targets()
        output = small_model(x, targets=targets)
        output["total_loss"].backward()

        backbone_params = list(small_model.backbone.parameters())
        assert len(backbone_params) > 0
        grads_found = sum(1 for p in backbone_params if p.grad is not None and p.grad.abs().sum() > 0)
        assert grads_found > 0, "No gradients flowed back to backbone parameters"


class TestYOLOXInference:
    """Tests for YOLOX in inference/eval mode."""

    def test_output_shape(self, small_model):
        small_model.eval()
        x = _dummy_input()
        with torch.no_grad():
            output = small_model(x)
        assert isinstance(output, torch.Tensor)
        assert output.dim() == 3
        assert output.shape[0] == BATCH_SIZE
        # Last dim should be 5 + num_classes (x, y, w, h, obj + cls scores)
        assert output.shape[2] == 5 + NUM_CLASSES

    def test_output_num_anchors_positive(self, small_model):
        small_model.eval()
        x = _dummy_input()
        with torch.no_grad():
            output = small_model(x)
        num_anchors = output.shape[1]
        assert num_anchors > 0, "Number of anchors should be positive"


MODEL_VARIANTS = [
    pytest.param(0.33, 0.25, True, id="yolox-nano"),
    pytest.param(0.33, 0.50, False, id="yolox-s"),
    pytest.param(0.67, 0.75, False, id="yolox-m"),
    pytest.param(1.0, 1.0, False, id="yolox-l"),
]


class TestYOLOXVariants:
    """Parametrized tests across model variants."""

    @pytest.mark.parametrize("depth,width,depthwise", MODEL_VARIANTS)
    def test_variant_inference(self, depth, width, depthwise):
        model = _build_model(depth, width, depthwise=depthwise)
        model.eval()
        x = _dummy_input()
        with torch.no_grad():
            output = model(x)
        assert output.shape[0] == BATCH_SIZE
        assert output.shape[2] == 5 + NUM_CLASSES

    @pytest.mark.parametrize("depth,width,depthwise", MODEL_VARIANTS)
    def test_variant_training(self, depth, width, depthwise):
        model = _build_model(depth, width, depthwise=depthwise)
        model.train()
        x = _dummy_input()
        targets = _dummy_targets()
        output = model(x, targets=targets)
        assert isinstance(output, dict)
        assert "total_loss" in output
        assert torch.isfinite(output["total_loss"]).item()


class TestYOLOXMisc:
    """Miscellaneous YOLOX tests."""

    def test_default_model_no_args(self):
        model = YOLOX()
        # Should create a valid model object even with None backbone/head
        assert model is not None

    def test_mode_switching(self, small_model):
        small_model.train()
        assert small_model.training is True

        small_model.eval()
        assert small_model.training is False

        small_model.train()
        assert small_model.training is True

    def test_inference_no_targets(self, small_model):
        """Calling forward without targets should produce inference output."""
        small_model.eval()
        x = _dummy_input()
        with torch.no_grad():
            output = small_model(x)
        assert isinstance(output, torch.Tensor)

    def test_training_then_eval_consistent(self, small_model):
        """Model should work in both modes within the same session."""
        small_model.train()
        x = _dummy_input()
        targets = _dummy_targets()
        train_out = small_model(x, targets=targets)
        assert isinstance(train_out, dict)

        small_model.eval()
        with torch.no_grad():
            eval_out = small_model(x)
        assert isinstance(eval_out, torch.Tensor)
