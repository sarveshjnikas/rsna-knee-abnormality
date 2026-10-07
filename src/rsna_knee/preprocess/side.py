"""Which knee (left or right) a study shows, from DICOM headers only.

Images are stored in one orientation for every series (header audit of all 24,371 training series), so
where "lateral" sits in the image depends only on the side: mirroring right knees puts every knee in the
same layout. Rule: the Laterality tag when present (50% of studies); otherwise the image centre's
left-right position in patient coordinates (+x is the patient's left), which agrees with the tag on
97.5% of the studies that have both.

    python -m rsna_knee.preprocess.side --series-csv <csv> --image-root <dir> --out knee_side.csv
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

MIN_OFFSET_MM = 5.0  # closer than this to the scanner's midline, geometry cannot tell


def series_vote(header) -> tuple[str | None, str | None]:
    """(side from the Laterality tag, side from geometry) for one series' header; None when unknown."""
    tag = str(getattr(header, "Laterality", "") or getattr(header, "ImageLaterality", "") or "").strip().upper()[:1]
    tag = tag if tag in ("L", "R") else None
    try:
        position = np.array(header.ImagePositionPatient, dtype=float)
        orientation = np.array(header.ImageOrientationPatient, dtype=float)
        spacing = np.array(header.PixelSpacing, dtype=float)
        centre_x = (position[0] + orientation[0] * spacing[1] * float(header.Columns) / 2
                    + orientation[3] * spacing[0] * float(header.Rows) / 2)
        geometry = None if abs(centre_x) < MIN_OFFSET_MM else ("R" if centre_x < 0 else "L")
    except (AttributeError, TypeError, ValueError, IndexError):
        geometry = None
    return tag, geometry


def study_side(votes: list[tuple[str | None, str | None]]) -> str | None:
    """Majority of the tags if any series has one, else majority of the geometry votes."""
    for kind in (0, 1):
        sides = [v[kind] for v in votes if v[kind]]
        if sides:
            return Counter(sides).most_common(1)[0][0]
    return None


def side_table(series_csv: Path, image_root: Path) -> pd.DataFrame:
    """One row per study: StudyInstanceUID, side (L / R / empty when unknown). Reads one header per series."""
    series = pd.read_csv(series_csv)
    rows = []
    for study, group in series.groupby("StudyInstanceUID"):
        votes = []
        for uid in group["SeriesInstanceUID"]:
            files = sorted((image_root / study / uid).glob("*.dcm"))
            if files:
                votes.append(series_vote(pydicom.dcmread(files[0], stop_before_pixels=True)))
        rows.append({"StudyInstanceUID": study, "side": study_side(votes)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--series-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    table = side_table(args.series_csv, args.image_root)
    table.to_csv(args.out, index=False)
    print(table["side"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
