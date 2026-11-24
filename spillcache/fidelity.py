"""Full-image fidelity metrics used by RegionE-style reference evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FidelityMetrics:
    psnr_db: float
    ssim: float
    lpips: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "psnr_db": None if math.isinf(self.psnr_db) else self.psnr_db,
            "ssim": self.ssim,
            "lpips": self.lpips,
        }


def _load_rgb_pair(candidate: str | Path, reference: str | Path) -> tuple[Any, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow and NumPy are required for fidelity metrics") from exc

    candidate_path = Path(candidate)
    reference_path = Path(reference)
    try:
        with Image.open(candidate_path) as image:
            candidate_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(reference_path) as image:
            reference_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"cannot decode fidelity input: {exc}") from exc
    if candidate_array.shape != reference_array.shape:
        raise ValueError(
            "fidelity metrics require equal image shapes; "
            f"candidate={candidate_array.shape}, reference={reference_array.shape}"
        )
    return candidate_array, reference_array


class FidelityEvaluator:
    """Reusable PSNR/SSIM evaluator with optional official AlexNet LPIPS."""

    def __init__(self, *, include_lpips: bool = False, device: str | None = None) -> None:
        self.include_lpips = bool(include_lpips)
        self.device = device
        self._lpips_model: Any | None = None
        self._lpips_calibration_path: Path | None = None

    @staticmethod
    def _package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _sha256(path: Path | None) -> str | None:
        if path is None or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _get_lpips_model(self) -> tuple[Any, Any]:
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise ImportError("torch and lpips are required when LPIPS is enabled") from exc
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self._lpips_model is None:
            self._lpips_model = (
                lpips.LPIPS(net="alex", version="0.1", verbose=False)
                .to(self.device)
                .eval()
            )
            module_path = getattr(lpips, "__file__", None)
            if module_path is not None:
                self._lpips_calibration_path = (
                    Path(module_path).resolve().parent / "weights" / "v0.1" / "alex.pth"
                )
        return self._lpips_model, torch

    def provenance(self) -> dict[str, Any]:
        if self.include_lpips and self._lpips_model is None:
            self._get_lpips_model()
        return {
            "psnr": {"data_range": 255.0},
            "ssim": {
                "implementation": "skimage.metrics.structural_similarity",
                "channel_axis": -1,
                "data_range": 255,
                "scikit_image": self._package_version("scikit-image"),
            },
            "lpips": {
                "enabled": self.include_lpips,
                "net": "alex" if self.include_lpips else None,
                "version": "0.1" if self.include_lpips else None,
                "device": self.device,
                "package": self._package_version("lpips"),
                "torch": self._package_version("torch"),
                "torchvision": self._package_version("torchvision"),
                "trunk_weights": (
                    "torchvision AlexNet IMAGENET1K_V1 (LPIPS default)"
                    if self.include_lpips
                    else None
                ),
                "calibration_path": (
                    str(self._lpips_calibration_path)
                    if self._lpips_calibration_path is not None
                    else None
                ),
                "calibration_sha256": self._sha256(self._lpips_calibration_path),
            },
            "numpy": self._package_version("numpy"),
            "pillow": self._package_version("Pillow"),
        }

    def evaluate(self, candidate: str | Path, reference: str | Path) -> FidelityMetrics:
        try:
            import numpy as np
            from skimage.metrics import structural_similarity
        except ImportError as exc:
            raise ImportError("NumPy and scikit-image are required for fidelity metrics") from exc

        candidate_array, reference_array = _load_rgb_pair(candidate, reference)
        difference = candidate_array.astype(np.float64) - reference_array.astype(np.float64)
        mse = float(np.square(difference).mean(dtype=np.float64))
        psnr_db = math.inf if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)
        ssim = float(
            structural_similarity(
                candidate_array,
                reference_array,
                channel_axis=-1,
                data_range=255,
            )
        )

        lpips_value: float | None = None
        if self.include_lpips:
            model, torch = self._get_lpips_model()
            left = torch.from_numpy(candidate_array.copy()).permute(2, 0, 1).unsqueeze(0)
            right = torch.from_numpy(reference_array.copy()).permute(2, 0, 1).unsqueeze(0)
            left = left.to(device=self.device, dtype=torch.float32) / 127.5 - 1.0
            right = right.to(device=self.device, dtype=torch.float32) / 127.5 - 1.0
            with torch.no_grad():
                lpips_value = float(model(left, right).item())

        return FidelityMetrics(psnr_db=psnr_db, ssim=ssim, lpips=lpips_value)
