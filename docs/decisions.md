# Decisions

Each entry: the choice, why, the evidence, and what would change it.

## D1. The 58 gold studies are for evaluation only, never training
- **Why:** 1.3% more training data is worthless; an honest yardstick is not.
- **Evidence:** 58/4,407 labelled. They are enriched (every one has ≥1 positive, Fracture 31%), so their prevalences say nothing about the test.
- **Would change if:** a larger gold set appeared.

## D2. Training targets come from the reports, read by an LLM
- **Why:** the only label source for the other 4,349 studies. An LLM handles negation, synonyms, severity and 9 languages; keywords do not.
- **Evidence:** host explicitly allows LLM APIs on report text. Forum benchmark against the 58: keyword rules 0.814 vs LLM 0.878 macro AUC.
- **Would change if:** the LLM labels scored no better than rules on gold-dev.

## D3. The labeller reproduces the annotators' definitions, with "not addressed" as its own answer
- **Why:** gold labels are image-based with strict thresholds (borderline = negative), so "mentioned in the report" is the wrong target. Silence means different things per finding (missing not at random).
- **Evidence:** host confirmed labels are image-based and authoritative. Reports vs gold disagree (EDA: "moderate effusion" → gold 0). Forum: silent Baker's ≈ absent, silent synovitis ≈ 34% positive.
- **How:** per label, the LLM outputs one of positive / uncertain / below_threshold / absent / not_addressed, plus a short evidence quote; a mapping turns states into probabilities.

## D13. Keep a severity tier ("below_threshold") instead of collapsing to negative
- **Why:** AUC rewards ranking. "Small effusion" and "no effusion" are both label 0 by definition, but the first is more often rated positive by the annotators, so merging them destroys ranking information.
- **Evidence (gold-dev 29):** v1 with a single "negative" scored 0.788 macro AUC. In v1, "negative" effusions were 36% gold-positive, and silence was 0% gold-positive for Baker's / Lateral OA / MCL / Contusion (ranked wrongly above "negative"). v2 with the split: **0.880**.
- **Mapping v2:** positive 0.90, uncertain 0.70, below_threshold 0.30, absent 0.05, not_addressed 0.05. Designed on dev only; test 29 untouched.

## D4. Soft labels with binary cross-entropy
- **Why:** BCE's minimum is at p = y, so a target of 0.3 teaches "30% likely", the honest answer for an unclear report. Hard 0/1 would inject confident errors.

## D5. Gold split: dev 29 for tuning, test 29 scored once
- **Why:** tuning the labeller on the studies it is scored on makes the score optimistic.
- **How:** multilabel-stratified halves, seed 0; every label has 4–18 positives per half.
- **Caveat:** with 29 studies, AUC gaps below ~0.02–0.03 are noise.

## D6. Cross-validation: 5 folds over the 4,349, grouped by site, stratified over all labels
- **Why:** validation must look like the test. Same-hospital studies in train and validation inflate scores.
- **Evidence:** contributing sites in Italy and Mongolia, yet no Italian or Mongolian training report → the test likely has unseen hospitals.
- **Site proxy:** no site column exists. Language where one hospital is credited for it (el, bg, de, nl); else report template with ≥20 reports; else the study is its own group. 29 site groups, largest 368 studies; folds 881–882 studies.
- **How:** greedy assignment, largest groups first, each to the fold with the least label/size imbalance (no library does grouped + multilabel).

## D7. Model inputs are "slots" chosen by plane × fat suppression
- **Why:** each finding is best seen in particular views (MCL on coronal, ACL on sagittal, bruises on fat-suppressed). The flags are given at test time, so no guessing.
- **Evidence:** fat-sat fluid series present in 100% (axial), 96.4% (coronal), 94.2% (sagittal) of studies. In train the two flags are always equal; 0/0 means "no fat suppression" (T1, PD or T2), not T1.
- **Would change if:** the test has flag combinations unseen in training; the code must not crash on them.

## D8. 2.5D (2D CNN over slices), not 3D
- **Why:** voxels are ~0.16 mm in-plane but ~3.3 mm between slices, so volumes are sharp in 2D and coarse across slices. 2D encoders come ImageNet-pretrained; 3D ones are heavier and pretrained on less.

## D9. Preprocess on Kaggle with the exact code used at inference
- **Why:** avoids moving 570 GB, and identical code for train and test removes train/test preprocessing skew.

## D10. Modest encoder; spend effort on labels
- **Evidence:** forum: scaling the encoder gained +0.0011; labels are the binding constraint.

## D11. Walking skeleton first
- **Why:** an end-to-end submission early surfaces packaging, runtime and offline-install problems while there is time to fix them.

## D12. Labeller model: OpenAI gpt-6.1-sol, low reasoning effort, Batch API
- **Why:** strong mid-tier model; key already available; Batch API halves the price.
- **Evidence:** dev 29 at standard price cost $0.13 (≈350 output tokens/report); full 4,407-report batch projected ≈ $10.
- **Engineering:** one request body for sync and batch; strict JSON schema (no free text to parse); every raw answer cached on disk keyed by study + prompt version, so re-runs are free and results never silently change.
