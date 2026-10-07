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
SLOTS_5 = SLOTS + ("cor_noFS",)  # v4 adds coronal non-fat-sat (MCL, compartment arthritis); first 4 unchanged
N_SLICES = 24  # slices sampled evenly across each series (v1)
IMAGE_SIZE = 256  # pixels per side after cropping and padding to square

# Preprocessing versions. A trained model records the version it was trained on, and inference
# preprocesses the test set with that same version.
#   v1: crop to the body (scale varies by study), slices across the whole stack
#   v2: fixed physical crop around the knee (same mm per pixel everywhere), central slices only
#   v3: v2 at higher resolution (320 px, 0.47 mm per pixel instead of 0.59)
#   v4: v2 framing, but slices about 3 mm apart (at most 32, never repeated; blanks after the real ones,
#       marked in a per-slice mask) and a 5th view, coronal non-fat-sat
PREPROCESS = {
    "v1": {"n_slices": 24, "image_size": 256, "crop_mm": None, "slice_range": (0.0, 1.0)},
    "v2": {"n_slices": 32, "image_size": 256, "crop_mm": 150.0, "slice_range": (0.06, 0.94)},
    "v3": {"n_slices": 32, "image_size": 320, "crop_mm": 150.0, "slice_range": (0.06, 0.94)},  # 0.47 mm/pixel
    "v4": {"n_slices": 32, "image_size": 256, "crop_mm": 150.0, "slice_range": (0.06, 0.94), "spacing_mm": 3.0,
           "slots": SLOTS_5},
}
