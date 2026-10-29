# Next experiments (from the Oct 7 analysis)

Every number below comes from our own runs on the 58 gold studies (radiologist labels, never trained on) or from
the training reports. Each experiment changes one thing and is judged on the leaderboard against the current best.

## What the analysis found

| # | Finding | Evidence |
|---|---|---|
| F1 | Five findings carry most of the error | R2 gold AUC: Synovitis 0.79, Lateral OA 0.86, Lateral Meniscus 0.86, PF OA 0.88, MCL 0.92; the other seven are 0.94–0.99 |
| F2 | For tears and wear, the report beats our image model; for fluid and edema, the image wins by far | Report vs model on gold: ACL 0.994 vs 0.975, MCL 0.947 vs 0.923, Lat. meniscus 0.875 vs 0.860; Effusion 0.818 vs 0.991, Fracture 0.801 vs 0.943, Contusion 0.815 vs 0.937 |
| F3 | One report/image mix for all findings is wrong in both directions | Best report weight chosen on gold dev: 0.4–0.8 for MCL, menisci, PF OA, synovitis; 0–0.1 for effusion, Baker's, contusion. Lat. meniscus 0.86 → 0.92 and MCL 0.92 → 0.97 as the report weight goes 0 → 0.6 (all 58). Per-finding choice on dev scored 0.921 on the untouched test half vs 0.918 for our fixed 0.3 |
| F4 | Synovitis is learned as "effusion" | Reports never mention synovitis in 85% of training studies (2% say absent). Model synovitis vs effusion rank correlation 0.93; truth 0.40. Among the 35 gold knees with effusion, synovitis AUC is only 0.63 |
| F5 | Arthritis is learned as "general wear", not per compartment | Model medial vs lateral OA correlation 0.88; truth 0.32. Gold lateral OA ranked by the model's *medial* OA prediction still scores 0.72 |
| F6 | Medial vs lateral sits on a different image side for left and right knees, and we never standardise it | Header audit: orientation is identical for all 24,371 series, so side alone decides where "lateral" is. 2,330 right / 2,077 left knees. Side known for every study (DICOM tag for 50%, geometry agrees with the tag 97.5%). We also flip images at random. Every strong public pipeline standardises side and does not flip |
| F7 | Uniform self-training is exhausted | Round 3 (0.3 report, R2 + v3 teachers), folds 0–1: gold 0.917 vs 0.921 for round 2, test half 0.898 vs 0.905 |
| F8 | ConvNeXt-Tiny (lr 1e-4) is the best second model | Fold 0: 0.921 alone (B0 0.920); B0 + ConvNeXt 0.925 (test half 0.905) vs B0 + v3 per-finding 0.924 (0.900). Gains on menisci (+0.04, +0.02) |
| F9 | Not worth more time | 24 epochs (flat), 320 px for B0 (flat), ResNet-50 after round 2 (adds nothing), more slices than 32 (62% of series have fewer than 32 real slices) |

## Experiments, in priority order

| ID | Change (one thing) | Why (finding) | Target findings | Cost | Measure |
|---|---|---|---|---|---|
| **X1** | ConvNeXt-Tiny family, 5 folds, lr 1e-4, round-2 labels; submit B0 + ConvNeXt | F8 | menisci, MCL | 5 folds (~3 h on 2 slots) | LB of B0 + ConvNeXt vs best B0-only |
| **X2** | Labels by finding type: report weight 0.5 for tears/wear/synovitis, 0.1 for effusion, Baker's, contusion, fracture; teacher = R2 (+ ConvNeXt when ready) | F2, F3, F7 | lat. meniscus, MCL, PF OA, synovitis | new label file (Mac, seconds) + 5 folds | LB vs R2 |
| **X3** | Standardise knee side (mirror right knees: coronal/axial left-right, sagittal slice order) and stop random flips (training and test-time) | F5, F6 | lateral OA, lateral meniscus, MCL | side table from DICOM headers (Kaggle CPU, minutes) + loader change + 5 folds | LB vs R2 |
| **X4** | Synovitis only learns from reports that mention it (loss masked where the report is silent) | F4 | synovitis | loader/loss change + 5 folds | LB vs R2; gold synovitis AUC |
| X5 | Silence is not "absent": per-finding value for "not mentioned" between absent (0.05) and below-threshold (0.30) | gold: silence is positive 14% of the time vs 3% for "absent" | synovitis, fracture, PF OA | new label file + 5 folds | low priority: self-training already overwrites most of it |

Winners combine at the end: best labels (X2/X4) × standard side (X3) × families (B0 + ConvNeXt), then a final ensemble.
