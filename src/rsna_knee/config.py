"""Project-wide constants: paths and the twelve target labels."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"  # competition files, exactly as downloaded
INTERIM = DATA / "interim"  # derived tables (site proxy, ...), rebuilt by code
SPLITS = DATA / "splits"  # gold split and CV folds, rebuilt from code + seed

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
