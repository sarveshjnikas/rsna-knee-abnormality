"""One DICOM series -> (N_SLICES, IMAGE_SIZE, IMAGE_SIZE) uint8 volume."""

from pathlib import Path

import cv2
import numpy as np
import pydicom

from rsna_knee.config import IMAGE_SIZE, N_SLICES

INTENSITY_PERCENTILES = (1.0, 99.5)


def sorted_slices(series_dir: Path) -> tuple[list[Path], np.ndarray | None]:
    """Slice files ordered by physical position along the slice normal (file names are random), and
    those positions in mm (None when the geometry tags are missing)."""
    paths = list(series_dir.glob("*.dcm"))
    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]
    try:
        orientation = np.array(headers[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        keys = [float(np.dot(normal, np.array(h.ImagePositionPatient, dtype=float))) for h in headers]
        positions = np.array(keys)
    except AttributeError:  # missing geometry tags: fall back to the acquisition order
        keys = [int(getattr(h, "InstanceNumber", 0)) for h in headers]
        positions = None
    order = np.argsort(keys, kind="stable")
    return [paths[i] for i in order], (positions[order] if positions is not None else None)


def sorted_slice_paths(series_dir: Path) -> list[Path]:
    return sorted_slices(series_dir)[0]


def pick_indices(n_total: int, n: int = N_SLICES, slice_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """n indices spread evenly over the chosen part of the stack (repeats when it is shorter than n)."""
    low, high = slice_range[0] * (n_total - 1), slice_range[1] * (n_total - 1)
    return np.linspace(low, high, n).round().astype(int)


def pick_by_mm(positions: np.ndarray, n: int, slice_range: tuple[float, float], spacing_mm: float) -> np.ndarray:
    """Slices about spacing_mm apart over the chosen part of the stack: at most n, never repeated.

    The gap widens beyond spacing_mm only when n slices at that gap would not cover the range.
    Stacks thicker than spacing_mm keep every slice in the range.
    """
    first, last = positions[0], positions[-1]
    lo, hi = first + slice_range[0] * (last - first), first + slice_range[1] * (last - first)
    step = max(spacing_mm, (hi - lo) / max(n - 1, 1))
    targets = np.arange(lo, hi + 1e-6, step)[:n] if step > 0 else np.array([lo])
    nearest = np.abs(positions[None, :] - targets[:, None]).argmin(axis=1)
    return np.unique(nearest)


def to_square(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Zero-pad to a centred square (keeps the aspect ratio), then resize."""
    h, w = image.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    padded[top : top + h, left : left + w] = image
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(padded, (size, size), interpolation=interpolation)


def crop_to_body(volume: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Crop all slices to one box around the body (from the max projection), dropping empty field of view."""
    projection = volume.max(axis=0)
    mask = projection > 0.1 * np.percentile(projection, 99)
    if not mask.any():
        return volume
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    pad_r, pad_c = int(margin * volume.shape[1]), int(margin * volume.shape[2])
    r0, r1 = max(rows[0] - pad_r, 0), min(rows[-1] + pad_r + 1, volume.shape[1])
    c0, c1 = max(cols[0] - pad_c, 0), min(cols[-1] + pad_c + 1, volume.shape[2])
    return volume[:, r0:r1, c0:c1]


def crop_physical(volume: np.ndarray, spacing: tuple[float, float], crop_mm: float) -> np.ndarray:
    """A crop_mm x crop_mm window centred on the body, so every study has the same mm per pixel.

    Areas of the window outside the image are zero-padded.
    """
    projection = volume.max(axis=0)
    mask = projection > 0.1 * np.percentile(projection, 99)
    rows, cols = (np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))) if mask.any() else (
        np.array([0, volume.shape[1] - 1]), np.array([0, volume.shape[2] - 1]))
    centre_r, centre_c = (rows[0] + rows[-1]) // 2, (cols[0] + cols[-1]) // 2
    half_r, half_c = int(round(crop_mm / 2 / spacing[0])), int(round(crop_mm / 2 / spacing[1]))
    out = np.zeros((volume.shape[0], 2 * half_r, 2 * half_c), dtype=volume.dtype)
    r0, c0 = centre_r - half_r, centre_c - half_c
    src_r0, src_c0 = max(r0, 0), max(c0, 0)
    src_r1, src_c1 = min(r0 + 2 * half_r, volume.shape[1]), min(c0 + 2 * half_c, volume.shape[2])
    out[:, src_r0 - r0 : src_r1 - r0, src_c0 - c0 : src_c1 - c0] = volume[:, src_r0:src_r1, src_c0:src_c1]
    return out


def normalise(volume: np.ndarray) -> np.ndarray:
    """Per-series percentile scaling to uint8: MRI intensities have no fixed units across scanners."""
    low, high = np.percentile(volume, INTENSITY_PERCENTILES)
    scaled = (volume - low) / max(high - low, 1e-6)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _frames(path: Path) -> np.ndarray:
    pixels = pydicom.dcmread(path).pixel_array.astype(np.float32)
    return pixels if pixels.ndim == 3 else pixels[None]  # multi-frame files hold several slices


def load_volume(series_dir: Path, n_slices: int = N_SLICES, size: int = IMAGE_SIZE, crop_mm: float | None = None,
                slice_range: tuple[float, float] = (0.0, 1.0), spacing_mm: float | None = None,
                unique: bool = False) -> np.ndarray:
    """(n_slices, size, size) uint8; with spacing_mm or unique, (k <= n_slices, size, size) with no repeated slice."""
    paths, positions = sorted_slices(series_dir)

    def pick(n_total: int) -> np.ndarray:
        if not spacing_mm:
            picked = pick_indices(n_total, n_slices, slice_range)
            return np.unique(picked) if unique else picked
        if positions is not None and len(positions) == n_total:
            return pick_by_mm(positions, n_slices, slice_range, spacing_mm)
        return np.unique(pick_indices(n_total, n_slices, slice_range))  # no geometry: by count, no repeats

    if len(paths) == 1:  # a single multi-frame file
        frames = _frames(paths[0])
        slices = [frames[i] for i in pick(len(frames))]
    else:
        slices = [_frames(paths[i])[0] for i in pick(len(paths))]
    shape = slices[0].shape  # rare series mix slice sizes; bring them to the first one's
    slices = [s if s.shape == shape else cv2.resize(s, shape[::-1]) for s in slices]
    spacing = getattr(pydicom.dcmread(paths[0], stop_before_pixels=True), "PixelSpacing", None)
    if crop_mm and spacing is not None:
        volume = crop_physical(np.stack(slices), (float(spacing[0]), float(spacing[1])), crop_mm)
    else:  # v1, or no pixel spacing in the header
        volume = crop_to_body(np.stack(slices))
    return normalise(np.stack([to_square(s, size) for s in volume]))
