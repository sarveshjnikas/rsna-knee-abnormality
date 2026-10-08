# Experiments

## Scoreboard (what worked)

Val = gold-58 macro AUC of the run's ensemble (models never train on the 58). LB = public leaderboard; *exp* = expected, val + 0.03 (calibrated on the first two submissions). Rank = place of that LB score on the leaderboard downloaded on the date shown (5,206 teams on Oct 5; bronze 0.943 ≈ 390, silver 0.945 ≈ 246, gold 0.958 ≈ 20; ~1,700 teams sit between 0.938 and 0.943).

| # | What changed | Val (gold-58) | LB | Rank (Oct 5) |
|---|---|---|---|---|
| E1a | Baseline: B0 attention, labels v3, 2 folds | 0.885 | **0.916** | 3,011 |
| E1 | Baseline, 5 folds | 0.896 (dev 0.911 / test 0.886) | **0.924** | 2,797 |
| A1 | ResNet-50 plain pooling, 1 fold | 0.878 | 0.908 *exp* | 3,195 |
| A1+ | E1 + ResNet-50 fold 0 | 0.900 | 0.930 *exp* | 2,642 |
| P2 | v2 data (150 mm crop, 32 central slices), B0 students, folds 0–1 | 0.913 (dev 0.926 / test **0.898**) | ~0.938 *exp* at 5 folds | – |
| P2 vs v1 | same folds 0–1, v1 students | 0.904 (dev 0.917 / test 0.888) | – | – |
| L4 | Public label sets vs ours (gold, labels only) | ours test 0.863; stevenleehans v4 0.879; pilkwang 0.862; lixin 0.827 | – | – |
| L4b | Same, after mixing 50/50 with our B0 image teacher | ours 0.907 vs ours+steven v4 0.906 (test) | – | – |
| A2 | ResNet-50 plain pooling students (labels r1), folds 0–1 | 0.898 (dev 0.914 / test 0.880) | – | – |
| A2+ | D1 + A2 averaged 50/50 by family | 0.911 (dev 0.924 / test **0.896**) | ~0.935 *exp* (test +0.006) | – |
| P2a | D1 + v2 students (folds 0–1), averaged by family | 0.914 (dev 0.927 / test 0.897) | **0.931** | **2,652** |
| D1 | Self-training round 1 students (0.5 LLM + 0.5 out-of-fold), 5 folds, flip TTA at inference | 0.908 (dev 0.922 / test 0.890) | **0.929** | **2,686** |
| P2 5f | v2 students (labels r1), 5 folds | 0.919 (dev 0.929 / test 0.900) | **0.934** | – |
| R2 | Self-training round 2 (0.3 LLM + 0.7 mean of v1+v2 students' OOF), v2, folds 0–1 | 0.921 (dev 0.931 / test **0.905**) vs r1 same folds 0.913 (test 0.898) | – | – |
| L24 | 24 epochs instead of 12 (v2, r1, fold 0) | 0.917 (dev 0.933 / test 0.896) vs 12 ep 0.915 (test 0.897) | – (not worth 2× time) | – |
| R2 5f | Round-2 students, all 5 folds | 0.922 (dev 0.933 / test 0.904) vs r1 0.919 (test 0.900) | **0.935** | – |
| V3 | v3 data (320 px, 0.47 mm/px), round-2 labels, fold 0 | shared pooling 0.921 (test 0.901), per-finding attention 0.919 (test 0.895) vs v2 0.920 (0.897) | – | – |
| V3+ | v2 fold 0 + v3 per-finding fold 0 averaged | 0.924 (test 0.902); rank corr 0.94 (v2 vs v3 shared: 0.97) | – | – |
| E2 | R2 + v3 per-finding (2 families, 10 models), family average | 0.923 (test 0.904) | **0.939** | – |
| I1 | Idea 1: blank repeated slices instead of feeding them twice (v2, R2 labels, 5 folds) | 0.919 (test 0.898); noise (R2 re-run, other seed, fold 0): 0.004 | 0.936 (+0.001) | – |
| I2 | Idea 2: + slices ~3 mm apart (v4 data, 4 views) | 0.922 (test 0.902) | 0.935 (-0.001 vs I1) | – |
| I3 | Idea 3: + 5th view, coronal non-fat-sat (v4, 5 views) | 0.915 (test 0.893) | 0.934 (-0.001 vs I2) | – |
| R3 | Round 3, uniform mix 0.3 (R2 + v3 teachers), folds 0-1 | 0.917 (test 0.898) vs R2 0.921 (0.905) | – (stopped) | – |
| X1 | ConvNeXt-Tiny lr 1e-4, 5 folds; B0 R2 + ConvNeXt (two families) | ConvNeXt 0.921 (test 0.904); pair 0.923 (test **0.907**) | **0.941** (+0.006 vs R2) | – |
| X2 | Labels by finding type: report 0.5 tears/wear/synovitis, 0.1 effusion/Baker's/contusion/fracture (B0, 5 folds) | 0.923 (test 0.909) | 0.936 (+0.001) | – |
| X3 | Standard knee side (right knees mirrored), no random flips (B0, 5 folds) | 0.922 (test 0.903) | 0.935 (±0) | – |
| X4 | Synovitis trained only where the report mentions it (B0, 5 folds) | 0.922 (test 0.907); synovitis-effusion coupling 0.97 -> 0.95 | waits for reset | – |
| A3 | ResNet-50 mean pooling on v2, round-2 labels, fold 0 | 0.905 (test 0.886) vs B0 0.920; B0+R50 0.917 < B0 alone; rank corr 0.92 (was 0.86) | dropped | – |

One change per row. CV = mean OOF macro AUC vs pseudo-labels (site-grouped 5-fold). Gold = macro AUC vs gold test 29 (looked at rarely).

## Labeller (macro AUC vs gold)

| ID | Date | Change | Dev 29 | Test 29 | Notes |
|---|---|---|---|---|---|
| L1 | Oct 4 | gpt-6.1-sol, prompt v1 (4 states), single mapping | 0.788 | – | silence ranked above "negative"; mild findings merged into negative |
| L2 | Oct 4 | prompt v2: + below_threshold, absent vs not_addressed; mapping v2 | 0.880 | – | Synovitis weakest (0.745): 18/29 not addressed |
| L3 | Oct 4 | mapping v3: silent synovitis from effusion (no new LLM calls) | 0.885 | **0.863** | Final labels. Test scored once on Oct 5; all 58: 0.872. Total LLM cost $8.68 |

## Model

| ID | Date | Change | CV | Gold | LB | Notes |
|---|---|---|---|---|---|---|
| E1a | Oct 5 | Baseline B0, labels v3: folds 0–1 of 5 (best epoch by fold val) | 0.872 (2 folds) | 0.885 (2-model avg; dev 0.896, test 0.876) | **0.916** | single folds 0.870 / 0.887; 2-model avg ≈ mean, little ensemble gain; weakest gold labels Synovitis 0.746, Lateral Meniscus 0.785 |
| E1b | Oct 5 | Same baseline, folds 0–3 | 0.869 (4 folds) | 0.897 (dev 0.916, test 0.887) | – | gold-58 reads ≈0.03 below LB (E1a: 0.885 gold → 0.916 LB) |
| E1 | Oct 5 | Baseline complete: B0 attention, labels v3, 5 folds | 0.865 | 0.898 (dev 0.913, test 0.889) | **0.924** | +0.001 over 4 folds |
| A1 | Oct 5 | ResNet-50 + plain mean pooling (no attention/transformer), fold 0 only | 0.859 (fold 0) | 0.878 (fold 0; B0 fold 0: 0.870) | – | preds correlate 0.857 with B0; B0+R50 fold-0 average 0.892 > two B0 folds 0.885; B0 5-fold + R50 fold 0: 0.900 → diversity pays |
| D0 | Oct 5 | Label-mix check on gold dev (no training): α·LLM + (1−α)·4-fold image prediction | – | dev: α=1 0.885, α=0.5 **0.933**, α=0 0.916 | – | the two sources err differently → basis for self-training (α = 0.5 for round 1) |
| S0 | Oct 5 | Smoke: fold 0, 2 epochs, shard 0 only (~1,180 train studies) | 0.757 (fold 0) | 0.736 (58) | – | loss 0.534 → 0.486; 46 GB GPU peak at batch 8; ~2–3 min/epoch |

## Queue (M4, one change at a time, ordered by expected gain per GPU-hour)

| ID | Change | Why it might help | Cost |
|---|---|---|---|
| E1 | Baseline: EfficientNet-B0, 4 slots, 24 × 256², labels v3 | reference point for everything below | 5 folds × ~2 h |
| E2 | + cor_noFS slot (5 slots) | anatomy view for MCL / OA / fractures; present in 77% | same |
| E3 | 320 px instead of 256 | menisci are thin; forum notes resolution matters for meniscus | ~1.6× time, re-preprocess |
| E4 | Flip TTA at inference | free, usually +0.002–0.005 | inference only |
| E5 | Larger encoder (EfficientNet-B2 / ConvNeXt-Nano) | forum says small gains; test last | ~2× time |
| E6 | Seed / backbone ensemble | variance reduction | linear in models |
| E7 | Per-label mapping fitted on gold dev | uses the dev half harder; risk of overfitting 29 studies | free |
