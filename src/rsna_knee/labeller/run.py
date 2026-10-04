"""Command line for the labeller.

    python -m rsna_knee.labeller.run gold --split dev   # label + score gold studies (sync)
    python -m rsna_knee.labeller.run batch              # Batch API, sequential chunks, until all cached
    python -m rsna_knee.labeller.run collect            # pull any completed logged batches into the cache
    python -m rsna_knee.labeller.run export             # write soft labels for every cached study
"""

import argparse

import pandas as pd

from rsna_knee.config import LABEL_DIR, LABELS, RAW, SPLITS
from rsna_knee.labeller import llm
from rsna_knee.labeller.evaluate import auc_table, status_vs_gold
from rsna_knee.labeller.mapping import soft_labels, status_table


def _spend(records: list[dict], price: tuple[float, float, float]) -> tuple[float, float]:
    """Total cost and mean output tokens of the given records."""
    usages = [r["usage"] for r in records if r.get("usage")]
    total = sum(llm.cost_usd(u, price) for u in usages)
    mean_out = sum(u["output_tokens"] for u in usages) / max(len(usages), 1)
    return total, mean_out


def gold(split: str) -> None:
    train = pd.read_csv(RAW / "train.csv")
    gold_split = pd.read_csv(SPLITS / "gold_split.csv")
    studies = train.merge(gold_split.loc[gold_split["split"] == split, ["StudyInstanceUID"]], on="StudyInstanceUID")

    llm.label_sync(studies)
    records = llm.load_cache()
    split_records = {uid: records[uid] for uid in studies["StudyInstanceUID"] if uid in records}

    statuses = status_table(split_records)
    print(f"\n{split}: {len(statuses)}/{len(studies)} studies parsed")
    print(auc_table(studies, soft_labels(statuses)).round(3).to_string())

    crosstab = status_vs_gold(studies, statuses)
    print("\nstatus vs gold (n, share gold-positive):")
    print(crosstab.pivot(index="label", columns="status", values="n").fillna(0).astype(int).to_string())

    cost, mean_out = _spend(list(split_records.values()), llm.PRICE_STANDARD)
    full_batch = cost / max(len(split_records), 1) * len(train) / 2
    print(f"\ncost {split}: ${cost:.3f} (standard price), mean output tokens {mean_out:.0f}")
    print(f"projected full run of {len(train)} reports with Batch API: ${full_batch:.2f}")


def export() -> None:
    statuses = status_table(llm.load_cache())
    out_status = LABEL_DIR / f"status_{llm.RUN_NAME}.csv"
    out_soft = LABEL_DIR / f"labels_{llm.RUN_NAME}.csv"
    statuses.to_csv(out_status, index=False)
    soft_labels(statuses).to_csv(out_soft, index=False)
    print(f"wrote {out_soft} and {out_status} ({len(statuses)} studies)")
    print((statuses[LABELS].apply(lambda c: c.value_counts(normalize=True)).T.round(3)).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gold_parser = sub.add_parser("gold")
    gold_parser.add_argument("--split", choices=["dev", "test"], default="dev")
    sub.add_parser("batch")
    sub.add_parser("collect")
    sub.add_parser("export")
    args = parser.parse_args()

    if args.command == "gold":
        gold(args.split)
    elif args.command == "batch":
        llm.run_batches(pd.read_csv(RAW / "train.csv"))
    elif args.command == "collect":
        llm.collect_batches()
    elif args.command == "export":
        export()


if __name__ == "__main__":
    main()
