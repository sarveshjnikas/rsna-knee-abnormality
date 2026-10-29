# RSNA Knee MRI Abnormality Detection

Multi-label detection of 12 knee abnormalities from MRI studies, built for the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition (Oct 2026).

**Final result: public LB 0.951, private LB 0.940 (macro AUC)**. Every model is our own: no public notebooks and no public blends.

![python](https://img.shields.io/badge/python-3.12-blue)
![pytorch](https://img.shields.io/badge/PyTorch-timm-ee4c2c)
![metric](https://img.shields.io/badge/private%20AUC-0.940-success)

---

## The problem

Each study is one patient's knee MRI: several DICOM series (sagittal, coronal and axial, with and without fat suppression), about 30 slices each. For each study the model outputs a probability for 12 findings:

| Group | Findings |
|---|---|
| Ligament tears | ACL, MCL |
| Meniscal tears | Medial meniscus, Lateral meniscus |
| Osteoarthritis | Medial, Lateral, Patellofemoral |
| Fluid / swelling | Effusion, Synovitis, Baker's cyst |
| Bone | Contusion, Acute fracture |

**The hard part is the labels.** Of 4,407 training studies, only **58** have image-based labels from radiologists. The other 4,349 come with only a free-text radiology report, in 9 languages. At test time there are images only. So the project is mainly about weak supervision: turning reports into training targets that agree with what radiologists see on the images.

## Approach

```mermaid
flowchart LR
    R[Radiology reports<br/>4,349 studies] --> L[LLM labeller<br/>5-state extraction]
    L --> M[State → soft label<br/>mapping]
    M --> ST{Self-training<br/>mix with teacher OOF}
    D[DICOM series] --> P[Preprocess<br/>canonical side, 140 mm crop,<br/>384 px, per-view slots]
    P --> W[Window model<br/>ConvNeXt-T over 3-slice windows<br/>+ per-finding attention]
    ST --> W
    W -- out-of-fold predictions --> ST
    W --> B[Rank blend of<br/>model families]
    B --> S[submission.csv]
```

### 1. Report labeller ([`src/rsna_knee/labeller`](src/rsna_knee/labeller))
- An LLM reads each report and returns one of five states per finding: `positive / uncertain / below_threshold / absent / not_addressed`, with a quoted evidence span. Output is strict JSON via the Batch API, with every answer cached on disk.
- **Why five states:** the labels are image-based with strict thresholds ("small effusion" counts as negative). For AUC, the ranking between "small effusion" and "no effusion" still matters, so mild findings get their own tier. This alone moved gold-dev AUC from **0.788 to 0.880**.
- The mapping from states to probabilities is fit on the dev half of the gold studies only.
- Final labeller: **0.872 macro AUC** on all 58 gold studies. Total API cost: **$8.68**.

### 2. Validation ([`src/rsna_knee/validation`](src/rsna_knee/validation))
- No hospital column exists, so a **site proxy** (report language and template, 29 groups) stands in for it. Folds are grouped by site and stratified across all 12 labels with a greedy assignment that works for soft labels.
- The 58 gold studies are never trained on. They are split into dev (29, for tuning) and test (29, scored rarely).

### 3. Preprocessing ([`src/rsna_knee/preprocess`](src/rsna_knee/preprocess))
- Picks one series per view slot: sag FS, cor FS, axial FS, sag non-FS, cor non-FS.
- **Canonical side:** left knees are mirrored into the right-knee layout. The sagittal stack runs lateral→medial for one side and medial→lateral for the other, so without this a slice-position embedding means opposite anatomy on the two sides. This was found in a review of error cases with an LLM agent prompted as an MSK radiologist ([`.claude/agents/radiologist.md`](.claude/agents/radiologist.md)).
- Fixed physical crop (140 mm → 384 px, about 0.36 mm/px), evenly spaced slices per view (26/30/30/22/20), per-series intensity normalisation.

### 4. Window model ([`src/rsna_knee/model`](src/rsna_knee/model))
- Each view is cut into **3-slice windows** fed to a 2D ImageNet backbone (ConvNeXt-Tiny), a 2.5D design.
- **Per-finding attention pooling** over all windows from all views: an ACL tear shows on 2 slices and osteoarthritis on many, so each finding gets its own attention. This is a form of multiple-instance learning.
- Masked BCE on soft labels, separate learning rates for backbone and head, an average of the final weights of the best 3 epochs, and 43 random windows per study in training with all 128 at test.

### 5. Self-training
Report labels and image-model predictions make *different* errors. On gold-dev, report labels alone score 0.885, image predictions alone 0.916, and a 50/50 mix **0.933**. Each round re-labels the training set with the current models' out-of-fold predictions, using per-finding weights: 0.5 for tears, OA and synovitis, 0.1 for effusion, cyst, contusion and fracture. A stronger teacher is used each round.

### 6. Blending ([`kaggle/blend`](kaggle/blend))
The metric only cares about ranking, so families are **rank-averaged**. Members are only added if they correlate clearly below 0.99 with the current blend.

## Results

Leaderboard progression (macro AUC, one change per submission; 40 scored submissions in total):

| Submission | Change | Public | Private |
|---|---|---|---|
| E1 | Baseline: EfficientNet-B0 2.5D attention, LLM labels, 5 folds | 0.924 | 0.911 |
| R2 | Self-training round 2 | 0.935 | 0.921 |
| X1 | + ConvNeXt-Tiny family | 0.941 | 0.927 |
| Y8 | Window ConvNeXt (3-slice windows, per-finding attention) | 0.948 | 0.935 |
| Y15 | + Canonical knee side, denser coronal/axial slices | 0.949 | 0.936 |
| Y19 | Rank blend of 3 families (different labels / side handling) | 0.950 | 0.937 |
| Y22 | Self-training round with window models as teacher | 0.951 | 0.938 |
| **Y29** | **Newest-label families only, rank blend** | **0.951** | **0.940** |

**What didn't help:** RadImageNet ResNet-50 (0.930), a 90 mm zoomed specialist (0.949), 5-fold instead of all-data seeds (0.947), more seeds, a transformer mixer across windows, and text-based OA severity re-grading.

Public-to-private drop: 0.011, so the model did not overfit the public LB.

## Post-competition analysis

The 1st-place team (public 0.967 / private 0.958) used the same backbone family, and their *single* ConvNeXt-T beat our whole ensemble by 0.014 private AUC. Their writeup points at the gap:

1. **Labels from images, not just text.** Their labelling agents viewed the slices together with the report (gold AUC about 0.93–0.94, against our text-only 0.872). We had concluded that "re-extraction can't help", which was only true for text-only re-extraction.
2. **Localisation supervision:** 3D ROI boxes per finding and per-slice lesion boxes, with features pooled inside each finding's box.
3. **Validation against a strong label key** across 20–30 LB submissions. Our 58-study gold set was too small to separate +0.003 changes.

Next experiment: train our window model on their published labels at identical settings. This separates the effect of the labels from the effect of the architecture.

## Lessons

- **Labels set the ceiling.** The one big jump (+0.011) came from labels, while model changes gave ≤ 0.001.
- **Diversity beats volume.** Blend members that agree 99% add nothing. Measure correlation before scaling a candidate.
- **A 58-study holdout can't resolve +0.003.** Fold-level gold gains under ~0.01 repeatedly failed to transfer to the LB.
- **Look at what the model sees.** The knee-side bug was found by looking at the images, not by tuning.

Full write-up: [docs/learnings.md](docs/learnings.md).

## Repository layout

```
src/rsna_knee/
  labeller/      LLM report extraction: prompt, schema, Batch API runner, state→label mapping, scoring
  validation/    site proxy, gold dev/test split, grouped multilabel folds
  preprocess/    DICOM → per-view slice stacks, canonical knee side
  model/         window dataset, WindowNet (per-finding attention), training, distillation, evaluation
  infer.py       test-time pipeline used inside the Kaggle notebook
kaggle/          kernels (preprocess, train, inference, audit, blend) and push scripts
notebooks/       exploratory data analysis
tests/           unit tests for folds, labeller, preprocessing, model and inference
docs/            experiment log, decisions with evidence, architecture, radiology consults
```

## Running it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Competition data goes in `data/raw/` (not included; redistribution is against the competition rules).

```bash
uv sync
uv run pytest
```

```bash
# 1. labels from reports (needs OPENAI_API_KEY in .env)
uv run python -m rsna_knee.labeller.run gold --split dev
uv run python -m rsna_knee.labeller.run batch
uv run python -m rsna_knee.labeller.run export

# 2. validation splits
uv run python -m rsna_knee.validation.sites
uv run python -m rsna_knee.validation.gold_split
uv run python -m rsna_knee.validation.folds

# 3. preprocess and train one fold (or --fold -1 for all data)
uv run python -m rsna_knee.preprocess.run --series-csv data/raw/train_series.csv \
    --image-root data/raw/train_series --out data/processed/train
uv run python -m rsna_knee.model.train_windows --fold 0 --processed data/processed/train --out runs/window

# 4. inference
uv run python -m rsna_knee.infer --comp-dir data/raw --models runs/window --work /tmp/test --out submission.csv
```

Training ran on one GPU per job (~8 min per fold epoch). Inference runs offline in a Kaggle notebook on 2×T4.

## Documentation

| Doc | Contents |
|---|---|
| [experiments.md](docs/experiments.md) | Every run, one change per row, with gold and LB scores |
| [decisions.md](docs/decisions.md) | Each design choice with its evidence and what would change it |
| [architecture.md](docs/architecture.md) | Pipeline and component overview |
| [learnings.md](docs/learnings.md) | What worked, what didn't, and why |
| [radiology/](docs/radiology) | Clinical description of the model's view, used for radiologist consults |
