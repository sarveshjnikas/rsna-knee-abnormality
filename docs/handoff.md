# RSNA Knee MRI: handoff for a fresh review (2026-10-15)

**Ask for the reviewer:** we have plateaued at public LB 0.949. Find what we are missing to reach **0.956** (the bronze line, also currently the silver line) by **Oct 22**.

**Constraints:**
- Our own models only; we may copy ideas, never public notebooks or blends.
- Every component must be explainable in an interview.

Read with: `docs/experiments.md` (full log), `docs/architecture.md`, `docs/decisions.md`, `docs/codex_reviews.md`, `docs/radiology/notes.md`.

## 1. Where we are

| Item | Value |
|---|---|
| Best public LB | 0.949 (Y15, Y16), rank ~1749 / 5957 |
| Bronze / silver | 0.956 |
| Gold | 0.962 |
| Public single models | ~0.949–0.950 |
| Public 0.956 notebooks | blends of 20–40 models |

**Compute:**
- Kaggle: 2×T4 GPUs, ~14 h left this week; quota resets Sat 00:00 UTC.
- University cluster: 2 jobs at a time, 1 GPU each, ~8 min per fold epoch.

## 2. Task and data

**Task:** multi-label abnormality detection per knee MRI study, scored by mean AUC over the findings.

**Findings:**
- ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA
- Contusion, Fracture, Synovitis
- the remaining competition labels (see `config.py`)

**Data:**
- Multi-series DICOM: sagittal, coronal and axial, with PD and FS sequences.
- About 570 GB raw, preprocessed into per-view slice stacks.
- `labels_x2`: labels derived from reports (LLM-extracted, two passes).
- `gold_split`: 58 studies with careful labels, used as our trusted holdout ("gold").
- 4-fold CV split. Validation AUC is ~0.964, which is optimistic because it uses noisy labels.

## 3. Model ("window model", `src/rsna_knee/model/`)

**Input:** each view is cut into slot windows of slices. In v7 the windows have `slot_slices (26, 30, 30, 22, 20)`, slice range 0.03–0.97, a 140 mm crop and 384 px.

**Architecture:**
- A 2D backbone (ConvNeXt-Tiny, ImageNet weights) encodes each window.
- Attention / top-k pooling (k=43) aggregates the windows into per-label logits (`network.py` WindowNet).

**Training:**
- 20 epochs, masked BCE, backbone lr 1e-4.
- Final weights are an average of the 3 best epochs.
- fp16 on T4.

**Canonical side:** left knees are mirrored into the right-knee layout, because the sagittal stack order depends on laterality. Verified on 6/6 cases.

**Inference:** the parity audit found a max diff of 0.003 between the submission path and training, and sides match 58/58. There is no test-time bug.

## 4. What we tried

**Leaderboard:**

| Run | Contents | LB |
|---|---|---|
| Y8 / Y11 / Y13 | Window ConvNeXt, v2/v5 preprocessing, blends | 0.948 |
| Y14 | v3 preprocessing, 4 seeds | 0.946 |
| Y15 | Canonical side + v7, 4 all-data seeds | 0.949 |
| Y16 | Y8 + Y15 families blended | 0.949 |
| Y17 (pending) | Y15 + v8 zoom specialist (joint findings only) | — |

**Gold-58, folds 0–1:**

| Idea | Gold AUC | Corr with base | Blend gain |
|---|---|---|---|
| Side + v7 base | 0.9334 (avg) | — | — |
| ConvNeXt-Small | 0.9303 | 0.985 | +0.0003 |
| r2 relabels | 0.9315 | 0.989 | +0.0012 |
| v7 without side | 0.9325 | 0.990 | +0.0013 |
| DINOv2 ViT-S | 0.9215 | — | negative |
| Transformer mixer across windows | trains after an lr fix | — | ~0 |
| Synovitis silent weight 0.25 | — | — | +0.002 (P 0.73) |

**Other results:**
- Side + v7 gave +0.005 on folds (P 0.84) but did not move the LB.
- Fold-level gold gains repeatedly failed to transfer to the LB.
- Longer training is ruled out: validation plateaus at epoch ~14–15 (e.g. fold 0 val 0.9636 → 0.9644 → 0.9637 at epochs 13 / 15 / 19).
- 9-epoch models are clearly weaker (gold 0.925 / 0.930).
- CoAtNet and ResNet50 distillation were tried earlier and were not better (see `experiments.md`).

**Diagnosis so far:** every model is the same window-CNN family, and members correlate about 0.99, so blending adds almost nothing.

## 5. Radiologist review (Opus agent, 12 error cases)

The 12 errors split into three groups of 4:
- **Fixable by viewing:** crop, resolution, slices skipped.
- **Visible but misread:** the finding is visible and the reader still gets it wrong.
- **Label noise.**

**Recommendations:**
- Near-native resolution on the joint.
- A patellofemoral zoom.
- Coronal views limited to the joint line.

The v8 run implements these.

## 6. In flight

**v8 region specialist:**
- 90 mm crop at 384 px (0.23 mm/px, near native).
- It votes only on the joint findings: ACL, MCL, both menisci, both OA, Contusion, Fracture.
- 2 all-data seeds, then auto-submitted as Y17.
- The LB result is expected late tonight or tomorrow, depending on the cluster queue.

**Fallback:** blend Y15 with wr2lab7_all and wv7_all.

## 7. Open questions for the reviewer

1. Why is the gold-fold → LB transfer so poor? Is gold-58 too small, or is the public LB subset different?
2. What does a structurally different model look like that we can train in 7 days on ~14 Kaggle GPU hours plus 2 cluster GPUs (2.5D/3D across slices, cross-view fusion)? Is it worth it?
3. Is per-finding weighting or a separate model for the weakest findings a better bet than whole-model changes?
4. Are we mishandling label noise? The labels come from reports, and a third of the errors are noise.
5. Anything in how we build the windows (slice selection, slot counts, crop centring) that caps AUC?
