"""Run RRDB ×4 inference after making every external dependency explicit.

The bundled network definition is usable without a checkpoint. Actual image
enhancement additionally needs an authorised compatible checkpoint, so this
module performs preflight checks before importing image-processing libraries or
allocating the full model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.rrdbnet_arch import RRDBNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_device(requested: str = "auto") -> torch.device:
    """Choose a requested compute device without silently forcing CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    return torch.device(requested)


def preflight(checkpoint: Path, input_directory: Path) -> list[str]:
    """Return actionable errors instead of failing later in image inference."""
    errors: list[str] = []
    if not checkpoint.is_file():
        errors.append(f"Checkpoint file not found: {checkpoint}")
    if not input_directory.is_dir():
        errors.append(f"Input directory not found: {input_directory}")
    elif not any(path.suffix.lower() in IMAGE_SUFFIXES for path in input_directory.iterdir()):
        errors.append(f"Input directory contains no supported image files: {input_directory}")
    return errors


def load_rrdbnet(checkpoint: Path, device: torch.device) -> RRDBNet:
    """Load the documented 23-block RRDB architecture from a state dictionary."""
    model = RRDBNet(3, 3, 64, 23, gc=32)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def enhance_directory(
    model: RRDBNet, input_directory: Path, output_directory: Path, device: torch.device
) -> int:
    """Enhance each readable RGB image and return the number of outputs written."""
    import cv2
    import numpy as np

    output_directory.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in input_directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    written = 0
    for path in images:
        bgr_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr_image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue
        rgb_tensor = torch.from_numpy(
            np.transpose(bgr_image[:, :, [2, 1, 0]], (2, 0, 1)).copy()
        ).float() / 255.0
        with torch.no_grad():
            result = model(rgb_tensor.unsqueeze(0).to(device)).squeeze(0).float().cpu().clamp(0, 1).numpy()
        output_bgr = np.transpose(result[[2, 1, 0]], (1, 2, 0)) * 255.0
        output_path = output_directory / f"{path.stem}_x4.png"
        if not cv2.imwrite(str(output_path), np.rint(output_bgr).astype(np.uint8)):
            raise OSError(f"Could not write output image: {output_path}")
        written += 1
    return written


def parse_arguments() -> argparse.Namespace:
    """Expose every path and device choice instead of relying on shell location."""
    parser = argparse.ArgumentParser(description="Run authorised ESRGAN/RRDB ×4 inference.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "RRDB_ESRGAN_x4.pth")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "samples")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    """Run preflight, load only a compatible checkpoint, then write ×4 outputs."""
    arguments = parse_arguments()
    errors = preflight(arguments.checkpoint, arguments.input_dir)
    if errors:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(errors))
    device = resolve_device(arguments.device)
    print(f"Loading RRDBNet checkpoint on {device.type}: {arguments.checkpoint}")
    model = load_rrdbnet(arguments.checkpoint, device)
    count = enhance_directory(model, arguments.input_dir, arguments.output_dir, device)
    print(f"Wrote {count} ×4 output image(s) to {arguments.output_dir}")


if __name__ == "__main__":
    main()
