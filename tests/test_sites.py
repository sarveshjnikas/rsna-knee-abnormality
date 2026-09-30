import pandas as pd

from rsna_knee.validation.sites import MIN_TEMPLATE_SIZE, assign_site_proxy, report_template


def test_template_ignores_side_dates_and_digits():
    left = report_template("[DATE]: MR Knie Links 15ch AA Klinische Inlichtingen")
    right = report_template("[DATE]: MR Knie Rechts 3ch AA Klinische Inlichtingen")
    assert left == right


def test_template_is_truncated():
    assert len(report_template("Exam Type: MRI KNEE RIGHT WO CONTRAST", n_chars=10)) == 10


def _studies(rows):
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "lang", "template"])


def test_single_site_language_groups_by_language_across_templates():
    studies = _studies([("a", "el", "tpl one"), ("b", "el", "tpl two")])
    assert assign_site_proxy(studies).tolist() == ["lang:el", "lang:el"]


def test_common_template_groups_by_template():
    rows = [(f"s{i}", "en", "exam type mri") for i in range(MIN_TEMPLATE_SIZE)]
    assert set(assign_site_proxy(_studies(rows))) == {"tpl:exam type mri"}


def test_rare_template_means_unknown_site():
    studies = _studies([("a", "en", "free text"), ("b", "en", "free text")])
    assert assign_site_proxy(studies).tolist() == ["study:a", "study:b"]
