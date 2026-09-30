"""Approximate which hospital each study comes from, using its report.

The data has no site column, but reports carry two strong fingerprints:
the language and the hospital's report template (its fixed opening words).

Grouping rule, from most to least confident:
1. Languages with a single credited contributing hospital (Greek, Bulgarian,
   German, Dutch) -> one group per language. Such a hospital uses several
   templates, so grouping by template would split it and leak.
2. Otherwise, a template shared by at least MIN_TEMPLATE_SIZE reports -> one
   group per template.
3. Otherwise the site is unknown -> the study is its own group, which puts no
   constraint on the folds.

Run as a module to write data/interim/site_proxy.csv:
    python -m rsna_knee.validation.sites
"""

import re

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

from rsna_knee.config import INTERIM, RAW

# One credited contributing hospital per language (see the competition overview).
SINGLE_SITE_LANGUAGES = {"el", "bg", "de", "nl"}

TEMPLATE_CHARS = 20
MIN_TEMPLATE_SIZE = 20

# Words for "left"/"right" vary between reports of the same template, so drop them.
_SIDE_WORDS = re.compile(
    r"\b(right|left|rt|lt|derech[oa]|izquierd[oa]|rechts|links|sağ|sol|"
    r"desn[aeiou]g?|lijev[aeiou]g?|droite?|gauche|δεξι\w*|αριστερ\w*|дяс\w*|ляв\w*)\b"
)


def detect_language(text: str) -> str:
    """ISO 639-1 language code of a report, or "unknown"."""
    DetectorFactory.seed = 0  # langdetect is random unless seeded
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def report_template(text: str, n_chars: int = TEMPLATE_CHARS) -> str:
    """Normalised opening of a report: lower case, no dates, sides, digits or punctuation."""
    text = text.strip().lower()
    text = re.sub(r"\[date\]|\[time\]", "", text)
    text = _SIDE_WORDS.sub("", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w]+", " ", text).strip()
    return text[:n_chars]


def assign_site_proxy(studies: pd.DataFrame) -> pd.Series:
    """Site group per study from its `lang` and `template` columns (see module docstring)."""
    template_size = studies["template"].map(studies["template"].value_counts())
    return pd.Series(
        [
            f"lang:{lang}" if lang in SINGLE_SITE_LANGUAGES
            else f"tpl:{template}" if size >= MIN_TEMPLATE_SIZE
            else f"study:{uid}"
            for uid, lang, template, size in zip(
                studies["StudyInstanceUID"], studies["lang"], studies["template"], template_size
            )
        ],
        index=studies.index,
        name="site_proxy",
    )


def build_site_table(train: pd.DataFrame) -> pd.DataFrame:
    """One row per study: language, template and site proxy."""
    table = train[["StudyInstanceUID"]].copy()
    table["lang"] = train["Report"].map(detect_language)
    table["template"] = train["Report"].map(report_template)
    table["site_proxy"] = assign_site_proxy(table)
    return table


def main() -> None:
    train = pd.read_csv(RAW / "train.csv")
    table = build_site_table(train)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / "site_proxy.csv"
    table.to_csv(out, index=False)

    kind = table["site_proxy"].str.split(":").str[0]
    print(f"wrote {out} ({len(table)} studies)")
    print(kind.value_counts().rename("studies").to_string())
    print(f"distinct site groups (excluding unknown): {table.loc[kind != 'study', 'site_proxy'].nunique()}")
    print(f"largest group: {table['site_proxy'].value_counts().iloc[0]} studies")


if __name__ == "__main__":
    main()
