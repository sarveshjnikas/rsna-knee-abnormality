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
N_SLICES = 24  # slices sampled evenly across each series
IMAGE_SIZE = 256  # pixels per side after cropping to the body and padding to square
