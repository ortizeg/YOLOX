import pytest
import torch

from yolox.models.yolo_pafpn import YOLOPAFPN


@pytest.fixture
def yolox_s():
    """YOLOX-S config: depth=0.33, width=0.50."""
    return YOLOPAFPN(depth=0.33, width=0.50)


@pytest.fixture
def yolox_l():
    """YOLOX-L config: depth=1.0, width=1.0."""
    return YOLOPAFPN(depth=1.0, width=1.0)


class TestOutputStructure:
    """Output is a tuple of 3 tensors."""

    def test_output_is_tuple_of_three_tensors(self, yolox_s):
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = yolox_s(x)
        assert isinstance(out, tuple)
        assert len(out) == 3
        for t in out:
            assert isinstance(t, torch.Tensor)


class TestYOLOXSShapes:
    """Correct shapes for yolox-s config (depth=0.33, width=0.50)."""

    def test_pan_out2_shape(self, yolox_s):
        x = torch.randn(2, 3, 640, 640)
        with torch.no_grad():
            pan_out2, _, _ = yolox_s(x)
        assert pan_out2.shape == (2, 128, 80, 80)

    def test_pan_out1_shape(self, yolox_s):
        x = torch.randn(2, 3, 640, 640)
        with torch.no_grad():
            _, pan_out1, _ = yolox_s(x)
        assert pan_out1.shape == (2, 256, 40, 40)

    def test_pan_out0_shape(self, yolox_s):
        x = torch.randn(2, 3, 640, 640)
        with torch.no_grad():
            _, _, pan_out0 = yolox_s(x)
        assert pan_out0.shape == (2, 512, 20, 20)


class TestYOLOXLShapes:
    """Correct shapes for yolox-l config (depth=1.0, width=1.0)."""

    def test_pan_out2_shape(self, yolox_l):
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            pan_out2, _, _ = yolox_l(x)
        assert pan_out2.shape == (1, 256, 80, 80)

    def test_pan_out1_shape(self, yolox_l):
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            _, pan_out1, _ = yolox_l(x)
        assert pan_out1.shape == (1, 512, 40, 40)

    def test_pan_out0_shape(self, yolox_l):
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            _, _, pan_out0 = yolox_l(x)
        assert pan_out0.shape == (1, 1024, 20, 20)


class TestDepthwiseVariant:
    """Depthwise variant produces same shapes as standard variant."""

    def test_depthwise_s_shapes_match_standard(self):
        model = YOLOPAFPN(depth=0.33, width=0.50, depthwise=True, act="silu")
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            pan_out2, pan_out1, pan_out0 = model(x)
        assert pan_out2.shape == (1, 128, 80, 80)
        assert pan_out1.shape == (1, 256, 40, 40)
        assert pan_out0.shape == (1, 512, 20, 20)

    def test_depthwise_l_shapes_match_standard(self):
        model = YOLOPAFPN(depth=1.0, width=1.0, depthwise=True, act="silu")
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            pan_out2, pan_out1, pan_out0 = model(x)
        assert pan_out2.shape == (1, 256, 80, 80)
        assert pan_out1.shape == (1, 512, 40, 40)
        assert pan_out0.shape == (1, 1024, 20, 20)


class TestGradientFlow:
    """Gradient flow to backbone parameters."""

    def test_gradients_reach_backbone(self, yolox_s):
        x = torch.randn(1, 3, 640, 640)
        out = yolox_s(x)
        loss = sum(o.sum() for o in out)
        loss.backward()

        backbone_params = list(yolox_s.backbone.parameters())
        assert len(backbone_params) > 0, "Backbone should have parameters"
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone_params)
        assert has_grad, "At least one backbone parameter should receive a gradient"

    def test_all_outputs_contribute_gradients(self, yolox_s):
        x = torch.randn(1, 3, 640, 640)
        out = yolox_s(x)

        # Use only one output at a time and verify gradients still flow
        for idx in range(3):
            yolox_s.zero_grad()
            out[idx].sum().backward(retain_graph=True)
            backbone_params = list(yolox_s.backbone.parameters())
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in backbone_params
            )
            assert has_grad, (
                f"Output index {idx} should propagate gradients to backbone"
            )


class TestDifferentInputSizes:
    """Different input sizes produce correctly scaled feature maps."""

    @pytest.mark.parametrize(
        "input_size, expected_strides",
        [
            (416, (52, 26, 13)),
            (320, (40, 20, 10)),
            (640, (80, 40, 20)),
        ],
    )
    def test_spatial_dimensions_scale_with_input(
        self, yolox_s, input_size, expected_strides
    ):
        x = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            pan_out2, pan_out1, pan_out0 = yolox_s(x)

        s8, s16, s32 = expected_strides
        assert pan_out2.shape[2:] == (s8, s8), f"Stride-8 output should be {s8}x{s8}"
        assert pan_out1.shape[2:] == (s16, s16), f"Stride-16 output should be {s16}x{s16}"
        assert pan_out0.shape[2:] == (s32, s32), f"Stride-32 output should be {s32}x{s32}"

    @pytest.mark.parametrize("input_size", [416, 320])
    def test_channel_dimensions_unchanged_across_input_sizes(
        self, yolox_s, input_size
    ):
        x = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            pan_out2, pan_out1, pan_out0 = yolox_s(x)

        assert pan_out2.shape[1] == 128
        assert pan_out1.shape[1] == 256
        assert pan_out0.shape[1] == 512
