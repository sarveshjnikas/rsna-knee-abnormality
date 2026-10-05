# Experiments

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
| E1 | Oct 5 | Baseline complete: B0 attention, labels v3, 5 folds | 0.865 | 0.898 (dev 0.913, test 0.889) | pending | +0.001 over 4 folds |
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
