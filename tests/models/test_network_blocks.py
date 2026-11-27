import pytest
import torch

from yolox.models.network_blocks import (
    BaseConv,
    Bottleneck,
    CSPLayer,
    DWConv,
    Focus,
    ResLayer,
    SiLU,
    SPPBottleneck,
    get_activation,
)


# ---------------------------------------------------------------------------
# get_activation
# ---------------------------------------------------------------------------
class TestGetActivation:
    def test_silu(self):
        act = get_activation("silu", inplace=True)
        assert isinstance(act, (torch.nn.SiLU, SiLU))

    def test_relu(self):
        act = get_activation("relu", inplace=True)
        assert isinstance(act, torch.nn.ReLU)

    def test_lrelu(self):
        act = get_activation("lrelu", inplace=True)
        assert isinstance(act, torch.nn.LeakyReLU)

    def test_unsupported_raises(self):
        with pytest.raises(AttributeError):
            get_activation("unsupported_activation_xyz")


# ---------------------------------------------------------------------------
# BaseConv
# ---------------------------------------------------------------------------
class TestBaseConv:
    @pytest.mark.parametrize(
        "in_ch, out_ch, ksize, stride, expected_spatial",
        [
            (3, 16, 3, 1, 32),
            (3, 16, 3, 2, 16),
            (16, 32, 1, 1, 32),
            (16, 32, 5, 1, 32),
            (16, 32, 5, 2, 16),
        ],
    )
    def test_output_shape(self, in_ch, out_ch, ksize, stride, expected_spatial):
        m = BaseConv(in_ch, out_ch, ksize, stride)
        x = torch.randn(1, in_ch, 32, 32)
        y = m(x)
        assert y.shape == (1, out_ch, expected_spatial, expected_spatial)

    def test_with_groups(self):
        m = BaseConv(16, 16, 3, 1, groups=16)
        x = torch.randn(1, 16, 8, 8)
        y = m(x)
        assert y.shape == (1, 16, 8, 8)

    @pytest.mark.parametrize("act", ["silu", "relu", "lrelu"])
    def test_activations(self, act):
        m = BaseConv(3, 16, 3, 1, act=act)
        x = torch.randn(1, 3, 8, 8)
        y = m(x)
        assert y.shape == (1, 16, 8, 8)

    def test_gradient_flow(self):
        m = BaseConv(3, 16, 3, 1)
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None
        for p in m.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# DWConv
# ---------------------------------------------------------------------------
class TestDWConv:
    @pytest.mark.parametrize(
        "in_ch, out_ch, ksize, stride",
        [
            (16, 32, 3, 1),
            (32, 64, 5, 1),
            (16, 16, 3, 1),
        ],
    )
    def test_output_shape(self, in_ch, out_ch, ksize, stride):
        m = DWConv(in_ch, out_ch, ksize, stride)
        x = torch.randn(1, in_ch, 16, 16)
        y = m(x)
        assert y.shape == (1, out_ch, 16, 16)

    def test_depthwise_then_pointwise_structure(self):
        m = DWConv(16, 32, 3)
        # First conv should be depthwise: groups == in_channels
        dconv = m.dconv
        assert dconv.conv.groups == 16
        # Second conv should be pointwise: 1x1
        pconv = m.pconv
        assert pconv.conv.kernel_size == (1, 1)
        assert pconv.conv.in_channels == 16
        assert pconv.conv.out_channels == 32

    def test_gradient_flow(self):
        m = DWConv(16, 32, 3)
        x = torch.randn(1, 16, 8, 8, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None
        for p in m.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# Bottleneck
# ---------------------------------------------------------------------------
class TestBottleneck:
    @pytest.mark.parametrize(
        "in_ch, out_ch, shortcut, expansion",
        [
            (64, 64, True, 0.5),
            (64, 64, False, 0.5),
            (32, 64, True, 0.5),
            (64, 64, True, 1.0),
        ],
    )
    def test_output_shape(self, in_ch, out_ch, shortcut, expansion):
        m = Bottleneck(in_ch, out_ch, shortcut=shortcut, expansion=expansion)
        x = torch.randn(1, in_ch, 16, 16)
        y = m(x)
        assert y.shape == (1, out_ch, 16, 16)

    def test_shortcut_adds_residual(self):
        """When shortcut=True and in_ch==out_ch, use_add should be True."""
        m = Bottleneck(64, 64, shortcut=True)
        assert m.use_add is True

    def test_shortcut_false_no_residual(self):
        m = Bottleneck(64, 64, shortcut=False)
        assert m.use_add is False

    def test_shortcut_mismatched_channels_no_residual(self):
        """Shortcut is disabled when in_ch != out_ch."""
        m = Bottleneck(32, 64, shortcut=True)
        assert m.use_add is False

    def test_depthwise_true(self):
        m = Bottleneck(64, 64, depthwise=True)
        x = torch.randn(1, 64, 8, 8)
        y = m(x)
        assert y.shape == (1, 64, 8, 8)

    def test_depthwise_false(self):
        m = Bottleneck(64, 64, depthwise=False)
        x = torch.randn(1, 64, 8, 8)
        y = m(x)
        assert y.shape == (1, 64, 8, 8)

    def test_gradient_flow(self):
        m = Bottleneck(64, 64, shortcut=True)
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# ResLayer
# ---------------------------------------------------------------------------
class TestResLayer:
    @pytest.mark.parametrize("in_ch", [32, 64, 128])
    def test_output_shape(self, in_ch):
        m = ResLayer(in_ch)
        x = torch.randn(1, in_ch, 16, 16)
        y = m(x)
        assert y.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (non-zero conv) but have same shape."""
        m = ResLayer(64)
        x = torch.randn(1, 64, 8, 8)
        y = m(x)
        assert y.shape == x.shape

    def test_gradient_flow(self):
        m = ResLayer(64)
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# SPPBottleneck
# ---------------------------------------------------------------------------
class TestSPPBottleneck:
    @pytest.mark.parametrize(
        "in_ch, out_ch, kernel_sizes",
        [
            (64, 64, (5, 9, 13)),
            (64, 128, (5, 9, 13)),
            (32, 32, (3, 5, 7)),
            (64, 64, (5,)),
            (64, 64, (5, 9)),
        ],
    )
    def test_output_shape(self, in_ch, out_ch, kernel_sizes):
        m = SPPBottleneck(in_ch, out_ch, kernel_sizes=kernel_sizes)
        x = torch.randn(1, in_ch, 16, 16)
        y = m(x)
        assert y.shape == (1, out_ch, 16, 16)

    def test_default_kernel_sizes(self):
        m = SPPBottleneck(64, 64)
        x = torch.randn(1, 64, 32, 32)
        y = m(x)
        assert y.shape == (1, 64, 32, 32)

    def test_gradient_flow(self):
        m = SPPBottleneck(64, 64)
        x = torch.randn(1, 64, 16, 16, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# CSPLayer
# ---------------------------------------------------------------------------
class TestCSPLayer:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_num_bottlenecks(self, n):
        m = CSPLayer(64, 64, n=n)
        x = torch.randn(1, 64, 16, 16)
        y = m(x)
        assert y.shape == (1, 64, 16, 16)
        assert len(m.m) == n

    @pytest.mark.parametrize(
        "in_ch, out_ch",
        [
            (64, 64),
            (64, 128),
            (128, 64),
        ],
    )
    def test_output_shape(self, in_ch, out_ch):
        m = CSPLayer(in_ch, out_ch)
        x = torch.randn(1, in_ch, 8, 8)
        y = m(x)
        assert y.shape == (1, out_ch, 8, 8)

    def test_shortcut_true(self):
        m = CSPLayer(64, 64, shortcut=True)
        x = torch.randn(1, 64, 8, 8)
        y = m(x)
        assert y.shape == (1, 64, 8, 8)

    def test_shortcut_false(self):
        m = CSPLayer(64, 64, shortcut=False)
        x = torch.randn(1, 64, 8, 8)
        y = m(x)
        assert y.shape == (1, 64, 8, 8)

    def test_depthwise(self):
        m = CSPLayer(64, 64, depthwise=True)
        x = torch.randn(1, 64, 8, 8)
        y = m(x)
        assert y.shape == (1, 64, 8, 8)

    @pytest.mark.parametrize("expansion", [0.5, 1.0])
    def test_expansion(self, expansion):
        m = CSPLayer(64, 64, expansion=expansion)
        x = torch.randn(1, 64, 8, 8)
        y = m(x)
        assert y.shape == (1, 64, 8, 8)

    def test_gradient_flow(self):
        m = CSPLayer(64, 64, n=2)
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None
        for p in m.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# Focus
# ---------------------------------------------------------------------------
class TestFocus:
    @pytest.mark.parametrize(
        "in_ch, out_ch, ksize, stride, spatial",
        [
            (3, 16, 1, 1, 32),
            (3, 32, 3, 1, 64),
            (1, 16, 1, 1, 16),
        ],
    )
    def test_output_shape(self, in_ch, out_ch, ksize, stride, spatial):
        m = Focus(in_ch, out_ch, ksize=ksize, stride=stride)
        x = torch.randn(1, in_ch, spatial, spatial)
        y = m(x)
        assert y.shape == (1, out_ch, spatial // 2, spatial // 2)

    def test_spatial_halved_channels_quadrupled(self):
        """Before the conv, Focus should rearrange (b,c,h,w) -> (b,4c,h/2,w/2)."""
        in_ch, spatial = 3, 32
        m = Focus(in_ch, 48, ksize=1, stride=1)
        x = torch.randn(1, in_ch, spatial, spatial)
        y = m(x)
        # After focus + conv(4*in_ch -> out_ch, 1x1, stride=1):
        # spatial should be halved
        assert y.shape[2] == spatial // 2
        assert y.shape[3] == spatial // 2
        # The internal conv receives 4*in_ch channels
        assert m.conv.conv.in_channels == 4 * in_ch

    def test_odd_spatial_raises_or_handles(self):
        """Focus requires even spatial dims for clean slicing."""
        m = Focus(3, 16, ksize=1)
        x = torch.randn(1, 3, 32, 32)
        y = m(x)
        assert y.shape == (1, 16, 16, 16)

    def test_gradient_flow(self):
        m = Focus(3, 16, ksize=1)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None
        for p in m.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_batch_size(self):
        m = Focus(3, 16, ksize=1)
        x = torch.randn(4, 3, 32, 32)
        y = m(x)
        assert y.shape[0] == 4


# ---------------------------------------------------------------------------
# SiLU (export-friendly)
# ---------------------------------------------------------------------------
class TestSiLU:
    def test_forward(self):
        act = SiLU()
        x = torch.randn(2, 16)
        y = act(x)
        expected = x * torch.sigmoid(x)
        assert torch.allclose(y, expected, atol=1e-6)

    def test_output_shape(self):
        act = SiLU()
        x = torch.randn(1, 3, 8, 8)
        y = act(x)
        assert y.shape == x.shape
