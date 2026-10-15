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

## X3 (Oct 13): severity grade for borderline lateral and PF OA

- **Why:** among borderline cases ("below threshold" or "uncertain"), about half of the gold cases are actually positive, so the old labels (0.30 / 0.70) barely separate them.
- **What:** I (Claude) graded the report's evidence snippet for each of the 2,778 borderline findings (2,054 unique snippets) on a 0–4 cartilage severity scale. 0 = grade I / superficial, 1 = mild / grade II / thinning, 2 = moderate / OA, unspecified / grade II–III, 3 = full-thickness / grade III–IV / high-grade / deep, 4 = advanced / bone-on-bone. Grades map to 0.10 / 0.25 / 0.50 / 0.80 / 0.90.
- **Mixed into X2:** the report weight for OA in X2 is about 0.5, so X3 = X2 + 0.5 × (new report label − old report label). File: `labels_x3_oagrade.csv`; 2,039 of 4,349 studies changed.
- **Synovitis:** not relabelled. 40 of 58 gold reports say nothing about synovitis and 14 of those are positive, so there is no text to fix.
- **Gold (report labels only, 58 studies):** lateral OA 0.821 → 0.843 (11 positive), PF OA 0.853 → 0.871 (21 positive). Within the borderline gold cases: 0.583 → 0.735 (33 cases).
- **Next:** train one image model on X3 and compare it on gold with the same model on X2.
- **ConvNeXt v3 (query head + view quota), folds 0–1, X2 labels:** gold 0.9336, against 0.9273 for v2 and 0.9338 for the query head alone. The two changes don't add up; the gain over v2 is mostly the query head. Queued `wcnxt3x3` (same recipe on X3 labels, folds 0–1) as the label test.
- **X3 result:** gold 0.9274 vs 0.9336 on X2, and worse on the targeted findings too (lateral OA 0.853 → 0.828, PF OA 0.876 → 0.849). Dropped. The image-based part of the labels already carried this signal; the text grade added noise.
- **Y12 (v3, 2 folds + 2 all-data seeds): LB 0.946**, vs 0.948 for Y8 (v2, 4 all-data seeds). Not like-for-like (fold models see 80% of the data). Next: Y13 = v2 + v3 as two families; 2 more v3 all-data seeds queued.

## Oct 13 evening: two like-for-like tests on the v2 recipe (folds 0–1, seed 0, baseline wcnxt2 folds 0–1 = gold 0.9273)
- **wsyn0 / wsyn25:** loss weight 0 or 0.25 on synovitis where the report is silent (silence is not a reliable no; gold: 14 of 40 silent reports are positive, and keywords in the text don't separate them).
- **wmix2 (window mixer):** 2 transformer layers over all windows before the per-finding pooling, so windows can cross-check each other across views (as a radiologist checks a spot on sagittal against coronal). Zero-initialised gate, so it starts identical to v2. Risk: trains on 32 windows, predicts on 96.
- **Y14 (v3: query head + view quota, 4 all-data seeds) = LB 0.946**, like-for-like with Y8 (v2) = 0.948. The +0.006 fold-gold gain did not transfer; 4-seed gold .932 vs .933. v3 dropped. Lesson: fold-gold gains under ~0.01 are noise.

## Oct 14 morning: night results (gold, folds 0–1; all / dev / test)
| Run | Gold all | Dev | Test | Lat. meniscus | Lat. OA | PF OA | Synovitis |
|---|---|---|---|---|---|---|---|
| v2 baseline (wcnxt2) | 0.9273 | 0.9459 | 0.9048 | 0.891 | 0.799 | 0.858 | 0.842 |
| Silent synovitis weight 0 | 0.9265 | 0.9447 | 0.9059 | 0.883 | 0.818 | 0.869 | 0.822 |
| Silent synovitis weight 0.25 | 0.9309 | 0.9468 | 0.9103 | 0.877 | 0.841 | 0.865 | 0.826 |
| Mixer, 32 windows | 0.8880 | 0.9114 | 0.8650 | | | | |
| Mixer, 48 windows | 0.6951 | 0.6999 | 0.6856 | | | | |
- **Synovitis:** synovitis itself got worse under both weights. The +0.0036 for weight 0.25 comes from lateral OA, which is below the 0.01 bar, so it's not a win. Replicating on folds 2–3 (wsyn25 f23).
- **Mixer:** the training collapsed (validation AUC 0.79 → 0.55 at epoch 3; baseline reached 0.935). Codex's diagnosis: the gate and the new transformer at head lr 1e-3 abruptly overwrite the backbone features. Fix: mixer and gate at backbone lr 1e-4 (wmix2lr, folds 0–1, k32). Hard stop if it doesn't come close to the baseline.

## Oct 14: radiologist consultation #1 and set 2
- **Radiologist finding:** the sagittal stack direction depends on the knee side. Lateral is at the end of the stack for right knees and at the start for left knees (verified on 6/6 cases against knee_side.csv). The model's slice-position embedding therefore means "lateral" for one side and "medial" for the other, and the mirror averaging at test time blends medial with lateral. The 4 weak findings are lateral, patellofemoral or synovitis, while the medial ones score 0.97–0.99.
- **Mixer retest (backbone lr):** trains normally now (fold-0 val 0.80 → 0.964). Gold 0.9268 vs 0.9273, so no gain. Mixer dropped.
- **Set 2, test 1 – canonical side (wside):** left knees mirrored into the right-knee layout in training and evaluation, no mirror averaging; v2 recipe, folds 0–1. Codex agreed it is the right first test.
- **Set 2, test 2 – v7 preprocessing (radiologist: too few slices cross the joint; the trim drops the suprapatellar recess):** cor_FS 18→30, axi_FS 18→30, cor_noFS 12→20 (128 windows), slice range 6%→3% trim, same 140 mm / 384 px. Train with k=43 (keeps the 32/96 per-window exposure; Codex). Run wcnxt2_v7: v2 recipe, folds 0–1, vs wcnxt2 on v5. Ranked first by expected gain (+0.002–0.005 est.).

## Oct 14: new goal (top 500 / bronze ~0.956, own models only) and the public-frontier read
- The medal lines (0.956) are public blends of 20–40 models from many teams; public single models are ~0.949–0.950, comparable to ours (0.948). Our gap is ensemble breadth of genuinely different strong families.
- **Set 3, test 1 – DINOv2 ViT-S/14 window family (wdino):** self-supervised pretraining (different from our ImageNet-supervised ConvNeXt/CoAtNet), same window recipe, backbone lr 2e-5, folds 0–1. Queued after the v7 unpack.
- **Canonical side, folds 0–1 (wside):** gold 0.9273 → 0.9313 (+0.004; paired bootstrap P(>0) = 0.90), dev +0.008, test flat. Lateral OA 0.799 → 0.820, PF OA 0.858 → 0.883, medial meniscus 0.966 → 0.980; lateral meniscus 0.891 → 0.876. Folds 2–3 pending.
- **Ops fix:** a slot sat empty ~40 min because v7 shards were unpacked one per job with 10-minute polling and DINOv2 was gated on the unpack. Now: one unpack job for shards 1–4, and both queues poll every 2 minutes.
- **Kaggle as a third GPU slot:** 2× T4, fp16 (T4 has no bf16; added `--amp fp16` with loss scaling). Speed test: 24 min to unpack v7, ~1 h per all-data epoch, 14.7 GB peak, so 2 models × 9 epochs per 12 h session. First session: side+v7+synovitis 0.25, seeds 0–1, 9 epochs (`rsna-knee-train-sv7syn`), as blend members; 8 h of quota kept for submissions.

## Oct 14 night: Y15 = 0.949
- **Y15 (side+v7, 4 all-data seeds): LB 0.949** vs 0.948. Gold for side+v7 folds 0–1 = 0.9327, the same as v7 alone (0.9325), so the side fix adds nothing on top of v7. The LB gain is smaller than gold suggested (+0.001 vs ~+0.005).
- **Codex:** most likely selection noise (gold winner's curse) plus LB rounding; possible test-time fault (side from headers, v7 at test). Highest-EV action: an end-to-end inference parity audit.
- **Parity audit (rsna-knee-audit):** the 58 gold train studies dressed as a test set through the exact submission path; compare with training-time predictions.
- **Replication queued:** side+v7 and v7 alone on folds 2–3.
- **Radiologist consultation #2:** blind-first, full 384 px, 9 dev-half error cases of side+v7.

## Oct 15: Y16 = 0.949; blending track
- **Y16 (Y8 v2 + Y15 side+v7, two families): LB 0.949**, same as Y15. The families correlate 0.991 on gold, so the blend adds little.
- **Parity audit:** the submission path matches training-time predictions (max diff 0.003, the fp16 vs bf16 gap), and sides match 58/58. No test-time bug.
- **All-data vs fold models:** side+v7 all-data gold 0.9328 vs v2 all-data 0.9334. The fold-level gain (+0.005 over 4 folds, P 0.84) does not survive at 100% data.
- **Kaggle 9-epoch models (side+v7+syn0.25):** gold 0.925 / 0.930 per seed, too weak (too few epochs).
- **Decision (user): blending.** Members must be strong (gold ≥ ~0.930) and different (correlation clearly < 0.99). Candidates on v7+side, folds 0–1: ConvNeXt-Small (wsmall7), CoAtNet window (wcoat7), ConvNeXt-T on r2 labels (wr2lab7).
