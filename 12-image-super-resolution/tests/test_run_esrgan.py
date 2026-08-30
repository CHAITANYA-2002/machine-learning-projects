from pathlib import Path

import torch

from src.rrdbnet_arch import RRDBNet
from src.run_esrgan import preflight, resolve_device


def test_rrdbnet_scales_a_small_rgb_tensor_by_four() -> None:
    """The network's two 2× upsampling stages define the advertised scale."""
    model = RRDBNet(in_nc=3, out_nc=3, nf=8, nb=1, gc=4)
    output = model(torch.rand(1, 3, 5, 7))

    assert output.shape == (1, 3, 20, 28)


def test_cpu_device_is_always_available() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_preflight_explains_missing_checkpoint_and_input_directory(tmp_path: Path) -> None:
    errors = preflight(tmp_path / "missing.pth", tmp_path / "missing-inputs")

    assert any("checkpoint" in error.lower() for error in errors)
    assert any("input" in error.lower() for error in errors)
