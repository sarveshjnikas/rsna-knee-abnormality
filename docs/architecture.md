# Architecture

Predict 12 knee-MRI findings per study. Metric: macro AUC over the 12 labels.
Train studies have reports; test studies have images only. Only 58 of 4,407 train studies carry real (image-based) labels.

## Pipeline

```
TRAIN                                                          TEST (Kaggle notebook, offline, <= 9 h)

reports (4,407) ──► LLM labeller ──► per-label mapping ──► soft labels (4,349)
                     (4 states +      (fit on gold-dev)            │
                      evidence)                                    ▼
DICOM ──► preprocess ──► series slots ──► 2.5D model ──► 12 logits ◄── BCE on soft labels
          sort slices      sag/cor/axi      2D CNN per slice,
          N slices         x fat-sat        attention over slices,
          resize, norm                      fusion over series
                                                   │
          site-grouped multilabel 5-fold CV ───────┤        DICOM ─► same preprocess ─► 5-fold ensemble ─► submission.csv
          gold 58: dev 29 (tune) / test 29 (once) ─┘
```

## Components

| # | Component | Code | Status |
|---|---|---|---|
| 1 | EDA | `notebooks/01_eda.ipynb` | ✅ |
| 2 | Validation: site proxy, gold split, folds | `src/rsna_knee/validation/` | ✅ (folds frozen once labels exist) |
| 3 | Report labeller | `src/rsna_knee/labeller/` | ⏳ |
| 4 | DICOM preprocessing | `src/rsna_knee/preprocess/` | ☐ |
| 5 | Model | `src/rsna_knee/model/` | ☐ |
| 6 | Training (GPU cluster) | `src/rsna_knee/train.py` | ☐ |
| 7 | Evaluation + ensembling | `src/rsna_knee/evaluate.py` | ☐ |
| 8 | Inference notebook | `kaggle/inference.ipynb` | ☐ |

Decisions and their evidence: [decisions.md](decisions.md). Scores: [experiments.md](experiments.md).

## Plan

| Milestone | Dates | Done when |
|---|---|---|
| M1 Labels | Oct 4–6 | Labeller scored on gold-dev; `labels_v1.csv` for 4,349 studies; folds frozen |
| M2 Data | Oct 5–7 | Preprocessing verified on the example study; Kaggle preprocessing run produces compact arrays |
| M3 Walking skeleton | Oct 7–9 | Fold-0 baseline trained; inference notebook submits; first LB score |
| M4 Improve | Oct 10–17 | One change at a time, logged: labels, slots, slices/resolution, encoder, augmentation; 5 folds; ensemble |
| M5 Final | Oct 18–20 | Final models, runtime check, 2 final submissions chosen |
| Buffer | Oct 21–22 | Final submission deadline Oct 22, 23:59 UTC |
