"""Turn the LLM's per-label status into soft training labels."""

import pandas as pd

from rsna_knee.config import LABELS
from rsna_knee.labeller.schema import Status, parse_findings

# Order matters for AUC. On gold-dev (v1): hedged findings were mostly positive, findings
# reported as mild were sometimes rated positive by the annotators, and silence was the
# strongest negative for most labels.
STATUS_PROBABILITY = {
    Status.POSITIVE: 0.90,
    Status.UNCERTAIN: 0.70,
    Status.BELOW_THRESHOLD: 0.30,
    Status.ABSENT: 0.05,
    Status.NOT_ADDRESSED: 0.05,
}


def status_table(records: dict[str, dict]) -> pd.DataFrame:
    """One row per study, one status column per label (unparseable answers are dropped)."""
    rows = []
    for uid, record in records.items():
        try:
            findings = parse_findings(record["output_text"])
        except ValueError:
            continue
        rows.append({"StudyInstanceUID": uid, **{label: findings[label].status.value for label in LABELS}})
    return pd.DataFrame(rows, columns=["StudyInstanceUID", *LABELS])


def soft_labels(statuses: pd.DataFrame, mapping: dict[Status, float] = STATUS_PROBABILITY) -> pd.DataFrame:
    probabilities = {status.value: p for status, p in mapping.items()}
    out = statuses[["StudyInstanceUID"]].copy()
    for label in LABELS:
        out[label] = statuses[label].map(probabilities)
    return out
