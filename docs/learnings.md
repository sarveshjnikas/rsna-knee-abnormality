# RSNA Knee: what we learnt

**Final:** public LB 0.951, the best of Y22, Y25, Y26, Y27, Y29 and Y30. Bronze was 0.956. All models are our own (ConvNeXt-T window models), with no public notebooks or blends.

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
3. **Diagnose before fixing.** A free status-vs-gold table showed the report labels are read correctly. Silent or vague reports are the limit, so re-extracting with a better prompt could not help.
4. **A 58-study holdout can't resolve +0.003.** Use out-of-fold predictions over thousands of studies, scored against one fixed key. Re-ranking every idea that way showed that the leaderboard plateau was real.
5. **Test one change at a time,** against its own base on the same folds.
6. **Submit early.** The leaderboard caught what gold missed.
7. **Look at what the model sees.** The knee-side issue was found by looking at images.
8. **Bring in a domain expert early.** The radiologist split the errors into thirds: viewing, misreading, label noise.
9. **Give outside agents a sanitized, read-only copy.** One cancelled a job and rewrote the queue.
10. **Never leave GPUs idle.** Queue the next job when you submit the current one, and test unattended scripts on the zero-jobs case: `grep -c` exits 1 on 0 matches.
11. **Forum claims are hypotheses.** Check them against your own evidence before acting on them.
12. **The medal is decided on the private LB.** Pick finals for robustness, not public score.

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
