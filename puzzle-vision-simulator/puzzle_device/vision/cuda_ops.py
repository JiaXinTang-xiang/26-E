#!/usr/bin/env python3
"""CUDA-accelerated OpenCV helpers for Jetson / NVIDIA GPU.

Transparent fallback to CPU when CUDA is not available, so the same code
runs on Windows, Linux-x86, and Jetson without any platform checks.

Design note — individual GPU ops have kernel-launch and upload/download
overhead that makes them *slower* than CPU for a single 1280×720 frame.
The win comes from *chaining* multiple ops on the GPU so the data stays
in device memory across the full pipeline.  Use the composite functions
(``segment_pieces_gpu``) rather than calling the one-off helpers below.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------

_cuda_checked: bool = False
_cuda_available: bool = False
_cuda_failure_reason: str | None = None

_REQUIRED_CUDA_APIS = (
    "createGaussianFilter",
    "cvtColor",
    "threshold",
    "createMorphologyFilter",
)


def _cuda_is_disabled_by_environment() -> bool:
    value = os.environ.get("PUZZLE_VISION_CUDA", "auto").strip().lower()
    return value in {"0", "false", "off", "no", "cpu"}


def _disable_cuda(reason: object) -> None:
    """Disable the optional GPU path after a capability or runtime failure."""
    global _cuda_checked, _cuda_available, _cuda_failure_reason
    _cuda_checked = True
    _cuda_available = False
    _cuda_failure_reason = str(reason)


def _check_cuda() -> bool:
    """Lazily detect the complete CUDA API required by this pipeline."""
    global _cuda_checked, _cuda_available, _cuda_failure_reason
    if _cuda_checked:
        return _cuda_available
    _cuda_checked = True
    if _cuda_is_disabled_by_environment():
        _cuda_failure_reason = "disabled by PUZZLE_VISION_CUDA"
        return False
    try:
        cuda_module = getattr(cv2, "cuda", None)
        gpu_mat_type = getattr(cv2, "cuda_GpuMat", None)
        if cuda_module is None or gpu_mat_type is None:
            raise AttributeError("OpenCV CUDA Python bindings are unavailable")
        missing = [name for name in _REQUIRED_CUDA_APIS if not hasattr(cuda_module, name)]
        if missing:
            raise AttributeError(f"missing OpenCV CUDA APIs: {', '.join(missing)}")
        if cuda_module.getCudaEnabledDeviceCount() <= 0:
            raise RuntimeError("no CUDA-enabled OpenCV device")
        cuda_module.setDevice(0)
        _ = gpu_mat_type(16, 16, cv2.CV_8UC1)
        _cuda_available = True
        _cuda_failure_reason = None
    except (cv2.error, AttributeError, RuntimeError, TypeError) as exc:
        _cuda_available = False
        _cuda_failure_reason = str(exc)
    return _cuda_available


def cuda_status() -> tuple[bool, str]:
    """Return whether the optional pipeline is usable and a short reason."""
    available = _check_cuda()
    return available, "available" if available else (_cuda_failure_reason or "unavailable")


# ---------------------------------------------------------------------------
# Composite pipeline — the recommended entry point
# ---------------------------------------------------------------------------


def segment_pieces_gpu(
    image: np.ndarray,
    background: np.ndarray | None,
    blur_size: int,
    color_distance_threshold: float | None,
    morph_size: int,
) -> np.ndarray:
    """CUDA-chained segmentation mask for the common background-subtraction path.

    Keeps data on the GPU for blur → colour-convert → threshold →
    morphology, downloading only the final binary mask.  Returns the same
    result as the CPU version. Actual performance depends on the OpenCV build,
    frame size, and transfer overhead, so callers must keep a CPU fallback.
    """
    if not _check_cuda():
        raise _Fallback

    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)

    # -- blur on GPU --------------------------------------------------------
    blur_type = cv2.CV_8UC3 if image.ndim == 3 else cv2.CV_8UC1
    blur_filter = cv2.cuda.createGaussianFilter(
        blur_type, blur_type, (blur_size, blur_size), 0,
    )
    gpu_blurred = blur_filter.apply(gpu_img)

    if background is not None and background.shape == image.shape:
        gpu_bg = cv2.cuda_GpuMat()
        gpu_bg.upload(background)
        gpu_ref = blur_filter.apply(gpu_bg)

        # BGR → LAB on GPU
        gpu_lab = cv2.cuda.cvtColor(gpu_blurred, cv2.COLOR_BGR2LAB)
        gpu_rlab = cv2.cuda.cvtColor(gpu_ref, cv2.COLOR_BGR2LAB)

        # Download LAB images for the per-pixel arithmetic (the CUDA Python
        # bindings don't reliably expose GpuMat scalar arithmetic).
        lab = gpu_lab.download().astype(np.float32)
        ref_lab = gpu_rlab.download().astype(np.float32)
        diff = lab - ref_lab
        distance = np.sqrt(np.mean(diff * diff, axis=2)).clip(0, 255).astype(np.uint8)
    else:
        # No-background path: do colour conversion on GPU, border-pixel
        # estimation on CPU (cheap), then upload the distance for threshold.
        gpu_lab = cv2.cuda.cvtColor(gpu_blurred, cv2.COLOR_BGR2LAB)
        lab = gpu_lab.download()
        h, w = lab.shape[:2]
        band = max(2, int(round(min(h, w) * 0.04)))
        border = np.concatenate([
            lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3),
            lab[band:-band, :band].reshape(-1, 3), lab[band:-band, -band:].reshape(-1, 3),
        ])
        bg_color = np.median(border, axis=0).astype(np.uint8)
        ref_bgr = cv2.cvtColor(bg_color.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]
        reference = np.empty_like(image)
        reference[:] = ref_bgr
        gpu_ref2 = cv2.cuda_GpuMat()
        gpu_ref2.upload(reference)
        gpu_rlab2 = cv2.cuda.cvtColor(
            blur_filter.apply(gpu_ref2), cv2.COLOR_BGR2LAB,
        )
        ref_lab2 = gpu_rlab2.download().astype(np.float32)
        diff = lab.astype(np.float32) - ref_lab2
        distance = np.sqrt(np.mean(diff * diff, axis=2)).clip(0, 255).astype(np.uint8)

    # -- threshold + morphology on GPU --------------------------------------
    gpu_dist = cv2.cuda_GpuMat()
    gpu_dist.upload(distance)

    if color_distance_threshold is None:
        # Otsu threshold — GPU doesn't support Otsu, do it on CPU then upload
        otsu_val, _ = cv2.threshold(
            distance, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        thresh = 8 if otsu_val < 8 else otsu_val
    else:
        thresh = float(color_distance_threshold)

    _, gpu_mask = cv2.cuda.threshold(
        gpu_dist, thresh, 255, cv2.THRESH_BINARY,
    )

    size = max(3, morph_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    gpu_mask = cv2.cuda.createMorphologyFilter(
        cv2.MORPH_OPEN, cv2.CV_8UC1, kernel,
    ).apply(gpu_mask)
    gpu_mask = cv2.cuda.createMorphologyFilter(
        cv2.MORPH_CLOSE, cv2.CV_8UC1, kernel,
    ).apply(gpu_mask)

    mask = gpu_mask.download()

    # Fill texture holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


class _Fallback(Exception):
    """Signal that the caller should fall back to the CPU path."""
