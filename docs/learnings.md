# RSNA Knee: what we learnt

**Final:** public LB 0.951 / **private 0.940** (Y29, Y30). The public bronze line was 0.956. All models are our own (ConvNeXt-T window models), with no public notebooks or blends. 1st place (Team Kangaroo) scored public 0.967 / private 0.958; their single ConvNeXt-T at 256 px scored 0.964 / 0.954.

## Post-mortem vs the winners (written 2026-10-29)

Source: the 1st-place write-up (kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution) and our `kaggle competitions submissions` table. Nothing below was re-run by us; "suspected" means the write-up suggests it but does not ablate it.

| Fact | Evidence |
|---|---|
| We did not overfit the public LB | Public to private drop: ours 0.011 (0.951 to 0.940), theirs 0.009 (0.967 to 0.958) |
| The backbone was not the gap | Their single model was the same family (ConvNeXt-T) and beat our whole ensemble by 0.014 private |
| Our labels were text-only | Report-text LLM labeller: gold-58 AUC 0.872, $8.68. Theirs: an agent per study viewed the slices plus the report: gold-58 AUC 0.93-0.94 |
| We had no localisation supervision | They had agents draw 3D ROI boxes (751 studies) and lesion boxes (111,029 boxes), trained a detector, cropped around the anatomy and pooled features inside each finding's box. We used a fixed 140 mm crop and 43 sampled windows |
| They used the report as a training target | Embedding-prediction head (Qwen3-Embedding-8B, loss weight 0.3), not used at inference. We used reports only to make labels |
| They calibrated soft labels | Temperature k about 0.77 fitted on gold-58 |
| Their ensemble was diverse, ours was not | MambaOut-B+, MaxxViT-V2-B, and 6-channel cross-series inputs. Ours: ConvNeXt-T windows, member correlation about 0.99 |
| Gold-58 could not see the gap | Their plain per-slice baseline scored 0.932 on gold-58 and ours 0.933, while the LB gap was 0.015 |
| Their selection metric was better | OOF AUC against stronger independent labels, checked against 20-30 public LB scores |
| Their architecture steps were small | +0.007 OOF AUC in their ablation; no gain from 2.5D, 3D or study-level models |

**Suspected causes, ranked (untested by us):** (1) image-aware labels, (2) box-based localisation, (3) report-embedding target, (4) label calibration, (5) ensemble diversity.
**Decisive test not done:** train our window model on the winners' public labels dataset and compare with our labels at the same settings.
**Correction to lesson 3 below:** the status-vs-gold table showed report text is read correctly, but the conclusion "re-labelling cannot help" held only for text-only labelling. The fix was to give the labeller the images.

## Score history (public LB)

| Step | LB | What moved it |
|---|---|---|
| First models (E1) | 0.924 | – |
| Self-training round 2 | 0.935 | Image teacher's out-of-fold predictions mixed into the report labels |
| Window ConvNeXt (Y8) | 0.948 | Per-finding attention over 3-slice windows, 20 epochs, 32 windows |
| Side + v7 (Y15) | 0.949 | Left knees mirrored to the right-knee layout, denser coronal/axial slices |
| 3-family rank blend (Y19) | 0.950 | Families trained on different labels and with/without side mirroring |
| X5 labels (Y22) | 0.951 | Self-training again, with the current window models as teacher |

**What didn't help:**

| Change | LB |
|---|---|
| v8 90 mm zoom specialist | 0.949 |
| RadImageNet ResNet-50 | 0.930 |
| ImageNet ResNet-50 | 0.946 |
| 5-fold instead of all-data seeds | 0.947 |
| More seeds | 0.950 |
| Synovitis 0.25 | 0.950 |
| X6/X7 rounds | 0.951 (no change) |

## Lessons
1. **Diversity beats volume.** Models that agree ~99% add nothing in a blend. Measure correlation before scaling a candidate.
2. **Labels set the ceiling.** Our one big jump (+0.011) came from labels, while model changes gave ≤ 0.001.
3. **Diagnose before fixing, then ask what information is missing.** A free status-vs-gold table showed the report labels are read correctly, and silent or vague reports are the limit of text. That ruled out re-extracting from text, but not adding the images to the labeller, which is what the winners did.
4. **A 58-study holdout can't resolve +0.003.** Use out-of-fold predictions over thousands of studies, scored against one fixed key. Re-ranking every idea that way showed that the leaderboard plateau was real.
5. **Test one change at a time,** against its own base on the same folds.
6. **Submit early.** The leaderboard caught what gold missed.
7. **Look at what the model sees.** The knee-side issue was found by looking at images.
8. **Bring in a domain expert early.** The radiologist split the errors into thirds: viewing, misreading, label noise.
9. **Give outside agents a sanitized, read-only copy.** One cancelled a job and rewrote the queue.
10. **Never leave GPUs idle.** Queue the next job when you submit the current one, and test unattended scripts on the zero-jobs case: `grep -c` exits 1 on 0 matches.
11. **Forum claims are hypotheses.** Check them against your own evidence before acting on them.
12. **The medal is decided on the private LB.** Pick finals for robustness, not public score.
13. **Calibrate the validation metric against the LB.** With 40+ submissions we could have checked which local metric tracked the LB. The winners did this with 20-30 submissions and chose OOF AUC against stronger independent labels.
14. **Supervise where the signal is.** Study-level labels leave the model hunting for a lesion. Box labels (ROI and lesion) let it pool features where the finding is.
15. **Use every text source as a target too.** The winners trained the model to predict a report embedding, which costs nothing at inference.
16. **Judge labels by log loss and confident errors,** not only AUC. Labels with similar AUC differed 2x in confident errors.

## ML concepts used
- **AUC:** depends only on the ranking within each finding. Rank-blend families so no model's probability scale dominates.
- **Weak, soft labels:**
  - LLM report extraction with a 5-status scheme (positive / uncertain / below threshold / absent / silent).
  - Self-training mixes the report labels with a teacher's out-of-fold predictions, with per-finding weights: 0.5 for tears, OA and synovitis; 0.1 for fluid, cyst, contusion and fracture.
- **2.5D windows + per-finding attention:** 2.5D windows feed a 2D pretrained backbone, and per-finding attention pooling is a form of multiple-instance learning. The 43 sampled training windows vs all 128 at test time made no difference (gold 0.9352 both ways).
- **Canonical side; fixed-mm crop:** mirror left knees into the right-knee layout; crop the same physical size from every study.
- **Seeds vs folds vs all-data:**
  - Seed averaging flattens after ~4 seeds.
  - All-data seeds beat 5-fold models here.
  - Averaging the weights of the last 3 epochs smooths the final model.
- **BatchNorm vs LayerNorm:** BatchNorm suffers when a batch is one study's correlated windows. Domain pretraining (RadImageNet) did not transfer here.
- **Winner's curse:** the best of many noisy comparisons overstates its gain.
