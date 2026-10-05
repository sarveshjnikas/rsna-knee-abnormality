"""Project-wide constants: paths and the twelve target labels."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"  # competition files, exactly as downloaded
INTERIM = DATA / "interim"  # derived tables (site proxy, ...), rebuilt by code
SPLITS = DATA / "splits"  # gold split and CV folds, rebuilt from code + seed
LABEL_DIR = DATA / "labels"  # LLM outputs (cached raw responses) and derived soft labels

# Column order matches train.csv and sample_submission.csv.
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Shape contract between preprocessing (writes it) and the model (reads it).
SLOTS = ("sag_FS", "cor_FS", "axi_FS", "sag_noFS")  # plane x sequence type, see preprocess/study.py
N_SLICES = 24  # slices sampled evenly across each series (v1)
IMAGE_SIZE = 256  # pixels per side after cropping and padding to square

# Preprocessing versions. A trained model records the version it was trained on, and inference
# preprocesses the test set with that same version.
#   v1: crop to the body (scale varies by study), slices across the whole stack
#   v2: fixed physical crop around the knee (same mm per pixel everywhere), central slices only
PREPROCESS = {
    "v1": {"n_slices": 24, "image_size": 256, "crop_mm": None, "slice_range": (0.0, 1.0)},
    "v2": {"n_slices": 32, "image_size": 256, "crop_mm": 150.0, "slice_range": (0.06, 0.94)},
}
