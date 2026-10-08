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
- **Frozen folds (Oct 5, labels v3):** 867–873 studies per fold; every label's share within a few points across folds.
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

## D14. Four input slots chosen by plane × fluid-sensitive flag; missing views are masked
- **Slots:** sag_FS, cor_FS, axi_FS, sag_noFS. FS = Fluid_Sensitive 1, so an unseen flag combination at test time still maps to a slot.
- **Evidence:** FS views present in 94–100% of studies; sag_noFS in 97%. cor_noFS (77%) and axi_noFS (19%) are left out of v1.
- **Missing:** a missing slot is zeros plus `present = False`; the model ignores it (tested: changing a missing view's pixels does not change the output).
- **Duplicates:** keep the series with the most slices (best coverage).

## D15. Preprocessing: 24 slices, crop to body, 256 px square, per-series intensity scaling
- **24 slices spread evenly** across each series: fixed shape for batching; median series has 30.
- **Crop to the body** (one box per series from the max projection): a 3D axial series is 640 × 1280 px with the knee small in the middle.
- **Pad to square, then resize to 256**: keeps the aspect ratio.
- **Per-series percentile scaling (1st–99.5th → 0–255):** MRI intensities have no fixed units across scanners.
- **Output:** uint8, compressed, ≈ 2.9 MB per study, ≈ 15 GB total.

## D16. Model: shared 2D encoder over 2.5D slices, attention over slices, fusion over views
- **2.5D input:** each slice with its two neighbours as the 3 RGB channels, so the ImageNet encoder gets some through-plane context without a 3D network.
- **Shared encoder** (EfficientNet-B0, ImageNet weights) for every slice of every view: one set of weights, 4 × 24 = 96 slices per study.
- **Attention pooling per view:** a finding sits in a few slices, so a learned weighting beats averaging.
- **Fusion:** view-type embedding + one transformer layer across views, then a masked mean and a 12-output head.

## D17. Training recipe
- BCE on soft targets; AdamW (lr 3e-4, weight decay 0.01); 1-epoch warm-up then cosine; bf16; gradient clip 1.0; 12 epochs; batch 8 studies.
- **Augmentation:** horizontal flip (labels are anatomical, a mirrored left knee is a right knee; QA of 1,469 preprocessed studies shows sagittal images facing left at some hospitals and right at others), zoom crop 0.85–1, per-view intensity gain/offset, view dropout 0.15 (test studies can lack views).
- **Checkpoint** chosen on the fold's validation AUC against pseudo-labels; the 58 gold studies are never trained on and are reported as an honest check.

## D18. Compute split: Kaggle CPU preprocesses, a GPU cluster trains, Kaggle GPU submits
- **Why:** the 570 GB of DICOMs already sit on Kaggle, so only the 15 GB of arrays move. Cluster A100s train. The submission must run as a Kaggle notebook anyway, with the same preprocessing code.

## D19. Labelling spend is capped and never paid twice
- Hard stop at $15 total (credit available: $19). Every answer is cached by study and prompt version; restarts first collect batches already in flight.
- OpenAI's Tier-1 queue holds 900k tokens for this model, so batches run as sequential chunks of 350 reports.

## D20. Silent synovitis inherits a share of the effusion probability
- **Why:** radiologists report effusion readily and synovitis rarely, and the two travel together. Silence about synovitis is therefore uninformative, not negative.
- **How:** only when the report does not address synovitis: p = 0.05 + 0.6 × p(effusion). An explicit synovitis statement is never overridden.
- **Evidence:** forum: P(synovitis | effusion) 0.63 vs 0.22 on gold. Gold dev: Synovitis AUC 0.745 → 0.806, macro 0.880 → 0.885 (decided before looking; the macro gain alone is within noise).

## D21. Self-training: blend report labels with out-of-fold image predictions
- **Why:** labels come from reports, the test is scored on image-based labels. A model that has seen thousands of knees predicts held-out studies closer to the radiologists than the reports do; report and image errors differ.
- **How:** new label = 0.5 × LLM label + 0.5 × out-of-fold prediction (the one fold model that never trained on the study). Gold studies are never relabelled. One α for all findings in round 1; per-finding α later.
- **Evidence:** gold dev, labels alone 0.885, 4-fold predictions alone 0.916, 50/50 mix 0.933. Student fold 0 at epoch 6: gold 0.899 vs teacher 0.875 at the same epoch (= the whole 5-fold teacher ensemble, 0.898).
- **Leaderboard (D1):** 0.929 vs 0.924 baseline, +0.005, much less than gold-58 overall suggested (+0.012). The untouched gold test half moved +0.004, matching the leaderboard; the dev half (+0.011) was inflated because α was chosen on it. Lesson: predict LB moves from gold **test**, not gold overall.
- **Caveat:** a student's fold-CV is against the blended labels, which were partly produced by teachers that trained on that fold's studies, so it is optimistic; compare students on gold.

## D22. A second architecture for ensemble diversity
- **What:** ResNet-50 with plain mean pooling over slices and views (no attention or transformer).
- **Evidence:** fold 0 gold 0.878 vs B0 0.870; predictions correlate 0.857 with B0; B0 + ResNet-50 on fold 0 averaged 0.892, above two B0 folds (0.885).

## D23. v2 preprocessing becomes the main data: fixed 150 mm crop, 32 central slices
- **Why:** with a body crop the scale varies by scanner field of view (one study's axial view used a 186 mm window, others 150 mm), so the same structure has a different size in pixels; a fixed physical window gives every study 0.59 mm/pixel. Central 6–94% of each stack drops skin-only edge slices.
- **Evidence:** same folds (0–1), same labels: gold test half 0.888 (v1) → 0.898 (v2); dev 0.917 → 0.926. Two v2 models beat the five-model v1 student ensemble on the test half.
- **Cost:** 33% more slices: 479 s per epoch instead of 360 s, 61 GB GPU memory instead of 46.

## D24. Public label sets are not added
- **Evidence:** alone, stevenleehans's v4 labels beat ours on the gold test half (0.879 vs 0.863). After the 50/50 mix with our out-of-fold image predictions, which is what students train on, there is no difference (0.906 vs 0.907).
- **Meaning:** the image teacher already supplies what better report reading would add. Gains now come from better teachers (better data, more model families), not from more labellers. Also avoids depending on third-party label licences.

## D25. Self-training round 2, and no longer training
- **Round 2:** label = 0.3 × LLM + 0.7 × mean out-of-fold prediction of the v1 and v2 round-1 students (two teachers, so the target leans on images more). Same folds 0–1 on v2: gold test half 0.898 → 0.905, all 58 0.913 → 0.921; both folds improved on their own. Folds 2–4 queued.
- **5 folds:** gold 0.922 (test half 0.904) vs round 1 0.919 (0.900).
- **ResNet-50 is dropped:** on round-2 labels it no longer adds diversity (prediction rank correlation with B0 0.92, was 0.86 in round 1; both students learn the same teacher). B0 + ResNet-50 on fold 0 scored 0.917 on gold, below B0 alone (0.920).
- **24 epochs:** fold 0, same labels: gold 0.915 → 0.917, test half flat (0.897 → 0.896). Not worth twice the GPU time; stay at 12.
- **Caveat (from a public write-up):** one team found mixed labels made fold models more alike and their 5-fold ensemble lost 0.001 on the LB. Our round 1 gained +0.005 on the LB, so we judge round 2 on the LB too.

## D26. Add our models to the public community stack as one fixed-weight leg
- **Why:** the bronze line (0.943) is one public notebook: a blend of ~30 public checkpoints (DINOv2, RadImageNet heads, CoAtNet at 336–384 px) that most of the 0.938–0.943 teams fork. An independently trained 0.929 model added at 15–30% scored 0.944 (three runs); 45% scored 0.942. Ours scores 0.934 alone.
- **How:** separate notebook (`kaggle/blend/build.py`): stack cells unchanged, then our 5 v2 models in a subprocess with our own decoders, rank-blended at a fixed 0.30 chosen before submitting. If our leg fails, the stack's submission is kept.
- **Risks:** the stack's constants were tuned on the public LB and the gold 58, so it may drop on private; our independent leg is the hedge. Runtime: the stack takes 6.5–8 h of 9.
- Our own pipeline stays separate and is still scored on its own.

## D27. Per-finding attention pooling (fusion="label")
- **Why:** a cruciate tear shows on two slices, osteoarthritis on many; one shared pooling makes them compete. Every (view, slice) is a token; each of the 12 findings has its own attention over all tokens and its own output. Used by the strongest public single models.
- **Result (v3, round-2 labels, fold 0):** alone it ties the shared pooling (gold 0.919 vs 0.921, within noise), but its predictions differ more (rank correlation with v2 0.94 vs 0.97), so v2 + v3 per-finding averages best (0.924). It becomes the second model family.
- **320 px alone did not help** (0.921 vs 0.920 at 256 px): resolution is not our bottleneck at B0 size.

## D28. Slice handling is not a lever; families are
- **Ideas 1-2 (one change each, leaderboard):** blank repeated slices 0.935 -> 0.936; slices ~3 mm apart 0.936 -> 0.935. Both inside leaderboard noise (one tick). They change how slices are fed, not what the model sees: 62% of series have fewer than 32 slices anyway, and mm spacing changes only thin-slice scans (~5%).
- **Idea 3 (5th view, coronal non-fat-sat):** 0.935 -> 0.934, gold 0.922 -> 0.915. Not used.
- **Second family (E2):** R2 + v3 per-finding 0.935 -> 0.939. **X1 (R2 + ConvNeXt-Tiny): 0.941**, our best own score. Averaging models that make different mistakes moved the score; next families: ConvNeXt (X1).
- **Round 3 with one uniform mix stopped:** gold 0.921 -> 0.917 on folds 0-1. Next label work changes the mix per finding type (X2) instead.

