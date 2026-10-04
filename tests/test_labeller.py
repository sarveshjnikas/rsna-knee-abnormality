import json

import pandas as pd
import pytest

from rsna_knee.config import LABELS
from rsna_knee.labeller.evaluate import auc_table
from rsna_knee.labeller.llm import cost_usd, output_text
from rsna_knee.labeller.mapping import soft_labels, status_table
from rsna_knee.labeller.schema import FIELD_OF_LABEL, JSON_SCHEMA, parse_findings


def _answer(status: str = "absent") -> str:
    return json.dumps({field: {"status": status, "evidence": ""} for field in FIELD_OF_LABEL.values()})


def test_schema_requires_every_field():
    assert set(JSON_SCHEMA["required"]) == set(FIELD_OF_LABEL.values())


def test_parse_maps_fields_to_competition_labels():
    findings = parse_findings(_answer("positive"))
    assert list(findings) == LABELS
    assert all(f.status.value == "positive" for f in findings.values())


def test_parse_rejects_missing_field():
    broken = json.loads(_answer())
    broken.pop("acl_tear")
    with pytest.raises(ValueError):
        parse_findings(json.dumps(broken))


def test_unparseable_answers_are_dropped():
    records = {"a": {"output_text": _answer()}, "b": {"output_text": "not json"}}
    assert status_table(records)["StudyInstanceUID"].tolist() == ["a"]


def test_soft_labels_keep_status_order():
    statuses = pd.DataFrame([{"StudyInstanceUID": s, **{l: s for l in LABELS}}
                             for s in ["positive", "uncertain", "below_threshold", "absent"]])
    probs = soft_labels(statuses)["ACL"].tolist()
    assert probs == sorted(probs, reverse=True)


def test_output_text_reads_message_parts():
    response = {"output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
    assert output_text(response) == "{}"


def test_cost_counts_cached_tokens_cheaper():
    usage = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 500}, "output_tokens": 100}
    assert cost_usd(usage, (2.0, 0.1, 10.0)) == pytest.approx((500 * 2.0 + 500 * 0.1 + 100 * 10.0) / 1e6)


def test_auc_perfect_ranking():
    gold = pd.DataFrame({"StudyInstanceUID": ["a", "b"], **{l: [1, 0] for l in LABELS}})
    pred = pd.DataFrame({"StudyInstanceUID": ["a", "b"], **{l: [0.9, 0.1] for l in LABELS}})
    assert auc_table(gold, pred)["macro"] == 1.0
