"""One DICOM series -> (N_SLICES, IMAGE_SIZE, IMAGE_SIZE) uint8 volume."""

from pathlib import Path

import cv2
import numpy as np
import pydicom

N_SLICES = 24
IMAGE_SIZE = 256
INTENSITY_PERCENTILES = (1.0, 99.5)


def sorted_slice_paths(series_dir: Path) -> list[Path]:
    """Slice files ordered by physical position along the slice normal (file names are random)."""
    paths = list(series_dir.glob("*.dcm"))
    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]
    try:
        orientation = np.array(headers[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        keys = [float(np.dot(normal, np.array(h.ImagePositionPatient, dtype=float))) for h in headers]
    except AttributeError:  # missing geometry tags: fall back to the acquisition order
        keys = [int(getattr(h, "InstanceNumber", 0)) for h in headers]
    return [paths[i] for i in np.argsort(keys, kind="stable")]


def pick_indices(n_total: int, n: int = N_SLICES) -> np.ndarray:
    """n indices spread evenly over the stack (repeats when the stack is shorter than n)."""
    return np.linspace(0, n_total - 1, n).round().astype(int)


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


def normalise(volume: np.ndarray) -> np.ndarray:
    """Per-series percentile scaling to uint8: MRI intensities have no fixed units across scanners."""
    low, high = np.percentile(volume, INTENSITY_PERCENTILES)
    scaled = (volume - low) / max(high - low, 1e-6)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _frames(path: Path) -> np.ndarray:
    pixels = pydicom.dcmread(path).pixel_array.astype(np.float32)
    return pixels if pixels.ndim == 3 else pixels[None]  # multi-frame files hold several slices


def load_volume(series_dir: Path, n_slices: int = N_SLICES, size: int = IMAGE_SIZE) -> np.ndarray:
    paths = sorted_slice_paths(series_dir)
    if len(paths) == 1:  # a single multi-frame file
        frames = _frames(paths[0])
        slices = [frames[i] for i in pick_indices(len(frames), n_slices)]
    else:
        slices = [_frames(paths[i])[0] for i in pick_indices(len(paths), n_slices)]
    shape = slices[0].shape  # rare series mix slice sizes; bring them to the first one's
    slices = [s if s.shape == shape else cv2.resize(s, shape[::-1]) for s in slices]
    volume = crop_to_body(np.stack(slices))
    return normalise(np.stack([to_square(s, size) for s in volume]))
