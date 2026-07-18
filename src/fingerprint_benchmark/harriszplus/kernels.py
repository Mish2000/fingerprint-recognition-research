"""Numerical kernels shared by the clean-room CPU and CUDA implementations."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def validate_grayscale_float32(image: np.ndarray, *, name: str = "image") -> np.ndarray:
    """Validate the detector's deliberately narrow image contract."""

    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if image.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional grayscale array.")
    if image.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32; got {image.dtype}.")
    if image.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not bool(np.isfinite(image).all()):
        raise ValueError(f"{name} must contain only finite values.")
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError(f"{name} values must lie in [0, 255]; got [{minimum}, {maximum}].")
    return np.ascontiguousarray(image)


def central_difference_numpy(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply mathematical ``[-1, 0, 1]`` differences with a zero frame."""

    source = validate_grayscale_float32(image)
    grad_x = np.zeros_like(source, dtype=np.float32)
    grad_y = np.zeros_like(source, dtype=np.float32)
    if source.shape[0] >= 3 and source.shape[1] >= 3:
        # Effective central difference: a left-to-right ramp has positive dx.
        grad_x[1:-1, 1:-1] = source[1:-1, 2:] - source[1:-1, :-2]
        grad_y[1:-1, 1:-1] = source[2:, 1:-1] - source[:-2, 1:-1]
    return grad_x, grad_y


def gaussian_kernel1d(sigma: float, *, truncate: float = 3.0) -> np.ndarray:
    """Return a normalized sampled Gaussian with radius ``ceil(3*sigma)``."""

    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    if not math.isfinite(truncate) or truncate <= 0.0:
        raise ValueError("truncate must be finite and positive.")
    unrounded_radius = truncate * sigma
    nearest_integer = round(unrounded_radius)
    if math.isclose(unrounded_radius, nearest_integer, rel_tol=1.0e-12, abs_tol=1.0e-12):
        radius = int(nearest_integer)
    else:
        radius = int(math.ceil(unrounded_radius))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    sigma32 = np.float32(sigma)
    kernel = np.exp(np.float32(-0.5) * (coordinates / sigma32) ** np.float32(2.0)).astype(
        np.float32,
        copy=False,
    )
    total = np.sum(kernel, dtype=np.float32)
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Gaussian kernel normalization failed.")
    kernel = (kernel / total).astype(np.float32, copy=False)
    # Correct the final float32 rounding residual without breaking symmetry.
    center = radius
    kernel[center] = np.float32(kernel[center] + (np.float32(1.0) - np.sum(kernel, dtype=np.float32)))
    return np.ascontiguousarray(kernel)


def gaussian_blur_numpy(
    image: np.ndarray,
    sigma: float,
    *,
    truncate: float = 3.0,
) -> np.ndarray:
    """Separable float32 Gaussian convolution using edge-including reflection."""

    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("gaussian_blur_numpy expects a two-dimensional array.")
    kernel = gaussian_kernel1d(sigma, truncate=truncate)
    output = cv2.sepFilter2D(
        source,
        ddepth=cv2.CV_32F,
        kernelX=kernel,
        kernelY=kernel,
        borderType=cv2.BORDER_REFLECT,
    )
    return np.ascontiguousarray(output, dtype=np.float32)


def sample_zscore_numpy(image: np.ndarray) -> np.ndarray:
    """Global z-score with sample standard deviation (ddof=1), flat to zero."""

    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("sample_zscore_numpy expects a two-dimensional array.")
    nonfinite_input_count = int(source.size - np.count_nonzero(np.isfinite(source)))
    if nonfinite_input_count:
        raise FloatingPointError(
            f"sample_zscore_numpy received {nonfinite_input_count} non-finite values."
        )
    if source.size <= 1:
        return np.zeros_like(source, dtype=np.float32)
    mean = np.mean(source, dtype=np.float32)
    centered = (source - mean).astype(np.float32, copy=False)
    variance = np.sum(centered * centered, dtype=np.float32) / np.float32(source.size - 1)
    standard_deviation = np.sqrt(np.float32(variance), dtype=np.float32)
    if not np.isfinite(standard_deviation):
        raise FloatingPointError("sample_zscore_numpy produced a non-finite standard deviation.")
    if standard_deviation <= 0.0:
        return np.zeros_like(source, dtype=np.float32)
    output = (centered / standard_deviation).astype(np.float32, copy=False)
    nonfinite_output_count = int(output.size - np.count_nonzero(np.isfinite(output)))
    if nonfinite_output_count:
        raise FloatingPointError(
            f"sample_zscore_numpy produced {nonfinite_output_count} non-finite values."
        )
    return output


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only in CPU-only installs.
        raise RuntimeError("The HarrisZ+ CUDA backend requires PyTorch.") from exc
    return torch


def central_difference_torch(image: Any) -> tuple[Any, Any]:
    """Torch counterpart of :func:`central_difference_numpy`."""

    torch = _require_torch()
    if image.ndim != 2 or image.dtype != torch.float32:
        raise ValueError("central_difference_torch expects a 2D float32 tensor.")
    grad_x = torch.zeros_like(image)
    grad_y = torch.zeros_like(image)
    if image.shape[0] >= 3 and image.shape[1] >= 3:
        grad_x[1:-1, 1:-1] = image[1:-1, 2:] - image[1:-1, :-2]
        grad_y[1:-1, 1:-1] = image[2:, 1:-1] - image[:-2, 1:-1]
    return grad_x, grad_y


def symmetric_indices_torch(length: int, pad: int, *, device: Any) -> Any:
    """Indices implementing OpenCV BORDER_REFLECT for arbitrary pad widths."""

    torch = _require_torch()
    if length <= 0 or pad < 0:
        raise ValueError("length must be positive and pad non-negative.")
    if pad == 0:
        return torch.arange(length, device=device, dtype=torch.int64)
    if length == 1:
        return torch.zeros(length + 2 * pad, device=device, dtype=torch.int64)
    coordinates = torch.arange(-pad, length + pad, device=device, dtype=torch.int64)
    period = 2 * length
    folded = torch.remainder(coordinates, period)
    return torch.where(folded < length, folded, period - folded - 1)


def symmetric_pad2d_torch(image: Any, pad_y: int, pad_x: int) -> Any:
    """Pad a 2D tensor exactly like OpenCV ``BORDER_REFLECT``."""

    if image.ndim != 2:
        raise ValueError("symmetric_pad2d_torch expects a two-dimensional tensor.")
    y_index = symmetric_indices_torch(image.shape[0], pad_y, device=image.device)
    x_index = symmetric_indices_torch(image.shape[1], pad_x, device=image.device)
    return image.index_select(0, y_index).index_select(1, x_index)


def gaussian_blur_torch(image: Any, sigma: float, *, truncate: float = 3.0) -> Any:
    """Separable float32 Gaussian convolution with BORDER_REFLECT padding."""

    torch = _require_torch()
    import torch.nn.functional as functional

    if image.ndim != 2 or image.dtype != torch.float32:
        raise ValueError("gaussian_blur_torch expects a 2D float32 tensor.")
    kernel_numpy = gaussian_kernel1d(sigma, truncate=truncate)
    kernel = torch.as_tensor(kernel_numpy, dtype=torch.float32, device=image.device)
    radius = int((kernel.numel() - 1) // 2)
    horizontal = symmetric_pad2d_torch(image, 0, radius)
    horizontal = functional.conv2d(
        horizontal[None, None],
        kernel.reshape(1, 1, 1, -1),
    )[0, 0]
    vertical = symmetric_pad2d_torch(horizontal, radius, 0)
    return functional.conv2d(
        vertical[None, None],
        kernel.reshape(1, 1, -1, 1),
    )[0, 0]


def sample_zscore_torch(image: Any) -> Any:
    """Torch sample z-score with an on-device flat-to-zero branch.

    Non-finite inputs or arithmetic remain non-finite so the detector's single
    per-scale response audit can count and reject them without adding several
    host synchronizations inside each z-score.
    """

    torch = _require_torch()
    if image.ndim != 2 or image.dtype != torch.float32:
        raise ValueError("sample_zscore_torch expects a 2D float32 tensor.")
    if image.numel() <= 1:
        return torch.zeros_like(image)
    mean = torch.mean(image)
    centered = image - mean
    variance = torch.sum(centered * centered) / float(image.numel() - 1)
    standard_deviation = torch.sqrt(variance)
    safe_standard_deviation = torch.where(
        standard_deviation == 0.0,
        torch.ones_like(standard_deviation),
        standard_deviation,
    )
    normalized = centered / safe_standard_deviation
    normalized = torch.where(
        standard_deviation == 0.0,
        torch.zeros_like(normalized),
        normalized,
    )
    return torch.where(
        torch.isfinite(standard_deviation),
        normalized,
        torch.full_like(normalized, float("nan")),
    )


# Compact aliases useful in focused mathematical tests.
central_difference = central_difference_numpy
gaussian_kernel = gaussian_kernel1d
gaussian_blur = gaussian_blur_numpy
sample_zscore = sample_zscore_numpy
