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
