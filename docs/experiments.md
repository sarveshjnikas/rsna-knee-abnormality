# Experiments

One change per row. CV = mean OOF macro AUC vs pseudo-labels (site-grouped 5-fold). Gold = macro AUC vs gold test 29 (looked at rarely).

## Labeller (macro AUC vs gold)

| ID | Date | Change | Dev 29 | Test 29 | Notes |
|---|---|---|---|---|---|
| L1 | Oct 4 | gpt-6.1-sol, prompt v1 (4 states), single mapping | 0.788 | – | silence ranked above "negative"; mild findings merged into negative |
| L2 | Oct 4 | prompt v2: + below_threshold, absent vs not_addressed; mapping v2 | 0.880 | – | Synovitis weakest (0.745): 18/29 not addressed |
| L3 | Oct 4 | mapping v3: silent synovitis from effusion (no new LLM calls) | 0.885 | – | Synovitis 0.745 → 0.806 |

## Model

| ID | Date | Change | CV | Gold | LB | Notes |
|---|---|---|---|---|---|---|
