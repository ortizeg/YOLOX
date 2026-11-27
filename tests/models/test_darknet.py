import pytest
import torch

from yolox.models.darknet import CSPDarknet, Darknet


# ---------------------------------------------------------------------------
# CSPDarknet tests
# ---------------------------------------------------------------------------

class TestCSPDarknet:
    """Tests for the CSPDarknet backbone."""

    def test_yolox_s_output_shapes(self):
        """YOLOX-s config (dep_mul=0.33, wid_mul=0.50) with 640x640 input."""
        model = CSPDarknet(dep_mul=0.33, wid_mul=0.50)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        out = model(x)

        # base_channels = int(0.50 * 64) = 32
        # channel progression: 32 -> 64 -> 128 -> 256 -> 512
        # spatial: 640 -> 320(stem) -> 160(dark2) -> 80(dark3) -> 40(dark4) -> 20(dark5)
        assert out["dark3"].shape == (1, 128, 80, 80)
        assert out["dark4"].shape == (1, 256, 40, 40)
        assert out["dark5"].shape == (1, 512, 20, 20)

    def test_yolox_l_output_shapes(self):
        """YOLOX-l config (dep_mul=1.0, wid_mul=1.0) with 640x640 input."""
        model = CSPDarknet(dep_mul=1.0, wid_mul=1.0)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        out = model(x)

        # base_channels = int(1.0 * 64) = 64
        # channel progression: 64 -> 128 -> 256 -> 512 -> 1024
        assert out["dark3"].shape == (1, 256, 80, 80)
        assert out["dark4"].shape == (1, 512, 40, 40)
        assert out["dark5"].shape == (1, 1024, 20, 20)

    def test_depthwise_output_shapes(self):
        """CSPDarknet with depthwise=True should produce identical shapes."""
        model = CSPDarknet(dep_mul=0.33, wid_mul=0.50, depthwise=True)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        out = model(x)

        assert out["dark3"].shape == (1, 128, 80, 80)
        assert out["dark4"].shape == (1, 256, 40, 40)
        assert out["dark5"].shape == (1, 512, 20, 20)

    @pytest.mark.parametrize(
        "out_features",
        [
            ("dark3",),
            ("dark4",),
            ("dark5",),
            ("dark3", "dark5"),
            ("dark3", "dark4"),
            ("dark4", "dark5"),
            ("dark3", "dark4", "dark5"),
        ],
    )
    def test_only_requested_out_features_returned(self, out_features):
        """Output dict must contain exactly the requested feature names."""
        model = CSPDarknet(dep_mul=0.33, wid_mul=0.50, out_features=out_features)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        out = model(x)

        assert set(out.keys()) == set(out_features)

    @pytest.mark.parametrize(
        "out_features",
        [
            ("dark3",),
            ("dark4", "dark5"),
            ("dark3", "dark4", "dark5"),
        ],
    )
    def test_gradient_flow_cspdarknet(self, out_features):
        """Gradients must flow back through every requested output stage."""
        model = CSPDarknet(dep_mul=0.33, wid_mul=0.50, out_features=out_features)
        model.train()
        x = torch.randn(1, 3, 640, 640, requires_grad=True)
        out = model(x)

        loss = sum(feat.sum() for feat in out.values())
        loss.backward()

        assert x.grad is not None
        assert torch.any(x.grad != 0)

    def test_batch_dimension(self):
        """Outputs must respect variable batch sizes."""
        model = CSPDarknet(dep_mul=0.33, wid_mul=0.50)
        model.eval()
        x = torch.randn(4, 3, 640, 640)
        out = model(x)

        for feat in out.values():
            assert feat.shape[0] == 4

    def test_different_input_size(self):
        """Spatial dimensions should scale proportionally with input size."""
        model = CSPDarknet(dep_mul=0.33, wid_mul=0.50)
        model.eval()
        x = torch.randn(1, 3, 416, 416)
        out = model(x)

        # 416 -> 208(stem) -> 104(dark2) -> 52(dark3) -> 26(dark4) -> 13(dark5)
        assert out["dark3"].shape == (1, 128, 52, 52)
        assert out["dark4"].shape == (1, 256, 26, 26)
        assert out["dark5"].shape == (1, 512, 13, 13)


# ---------------------------------------------------------------------------
# Darknet tests
# ---------------------------------------------------------------------------

class TestDarknet:
    """Tests for the Darknet (Darknet-53 / Darknet-21) backbone."""

    def test_darknet53_output_shapes(self):
        """Darknet depth=53 with 640x640 input."""
        model = Darknet(depth=53)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        out = model(x)

        # Channel progression: 32 -> 64 -> 128 -> 256 -> 512 -> (dark5 SPP ends at 512)
        # Spatial: 640 -> 320(stem) -> 160(dark2) -> 80(dark3) -> 40(dark4) -> 20(dark5)
        assert out["dark3"].shape == (1, 256, 80, 80)
        assert out["dark4"].shape == (1, 512, 40, 40)
        # dark5 ends with SPP block that reduces back to 512 channels
        assert out["dark5"].shape == (1, 512, 20, 20)

    def test_darknet21_output_shapes(self):
        """Darknet depth=21 with 640x640 input."""
        model = Darknet(depth=21)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        out = model(x)

        assert out["dark3"].shape == (1, 256, 80, 80)
        assert out["dark4"].shape == (1, 512, 40, 40)
        assert out["dark5"].shape == (1, 512, 20, 20)

    @pytest.mark.parametrize(
        "out_features",
        [
            ("dark3",),
            ("dark5",),
            ("dark3", "dark5"),
            ("dark3", "dark4", "dark5"),
        ],
    )
    def test_only_requested_out_features_returned(self, out_features):
        """Output dict must contain exactly the requested feature names."""
        model = Darknet(depth=53, out_features=out_features)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        out = model(x)

        assert set(out.keys()) == set(out_features)

    @pytest.mark.parametrize(
        "out_features",
        [
            ("dark3",),
            ("dark4", "dark5"),
            ("dark3", "dark4", "dark5"),
        ],
    )
    def test_gradient_flow_darknet(self, out_features):
        """Gradients must flow back through every requested output stage."""
        model = Darknet(depth=53, out_features=out_features)
        model.train()
        x = torch.randn(1, 3, 640, 640, requires_grad=True)
        out = model(x)

        loss = sum(feat.sum() for feat in out.values())
        loss.backward()

        assert x.grad is not None
        assert torch.any(x.grad != 0)

    def test_batch_dimension(self):
        """Outputs must respect variable batch sizes."""
        model = Darknet(depth=53)
        model.eval()
        x = torch.randn(2, 3, 640, 640)
        out = model(x)

        for feat in out.values():
            assert feat.shape[0] == 2
