"""The structured answer the LLM must return for every report."""

from enum import Enum

from pydantic import BaseModel, create_model

from rsna_knee.config import LABELS


class Status(str, Enum):
    POSITIVE = "positive"
    UNCERTAIN = "uncertain"
    BELOW_THRESHOLD = "below_threshold"
    ABSENT = "absent"
    NOT_ADDRESSED = "not_addressed"


# JSON field name the LLM fills, per competition label (same order as LABELS).
FIELD_OF_LABEL = {
    "ACL": "acl_tear",
    "MCL": "mcl_tear",
    "Medial Meniscus": "medial_meniscus_tear",
    "Lateral Meniscus": "lateral_meniscus_tear",
    "Medial OA": "medial_compartment_oa",
    "Lateral OA": "lateral_compartment_oa",
    "PF OA": "patellofemoral_oa",
    "Effusion": "joint_effusion",
    "Synovitis": "synovitis",
    "Baker's": "bakers_cyst",
    "Contusion": "bone_contusion",
    "Fracture": "acute_fracture",
}
assert list(FIELD_OF_LABEL) == LABELS


class Finding(BaseModel):
    status: Status
    evidence: str


ReportFindings = create_model("ReportFindings", **{field: (Finding, ...) for field in FIELD_OF_LABEL.values()})

# Written out by hand (no $refs) so it is valid for strict structured outputs.
_FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": [s.value for s in Status]},
        "evidence": {"type": "string"},
    },
    "required": ["status", "evidence"],
    "additionalProperties": False,
}
JSON_SCHEMA = {
    "type": "object",
    "properties": {field: _FINDING_SCHEMA for field in FIELD_OF_LABEL.values()},
    "required": list(FIELD_OF_LABEL.values()),
    "additionalProperties": False,
}


def parse_findings(text: str) -> dict[str, Finding]:
    """Validate the LLM's JSON and return {competition label: Finding}."""
    parsed = ReportFindings.model_validate_json(text)
    return {label: getattr(parsed, field) for label, field in FIELD_OF_LABEL.items()}
