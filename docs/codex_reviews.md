# Codex second opinions

Codex CLI, read-only, on a copy of `src/` and `docs/` only. Each entry gives the question, Codex's points, and our decision.

## 2026-10-13 21:30: review of the mixer and synovitis tests, and the best bets for +0.003
| Codex point | Our decision |
|---|---|
| `masked_bce` with weights still divides by the count of known cells, so the total loss scale drops | Kept. Adam is nearly scale-invariant, and the other 11 findings keep their weight |
| Mixer trains on 32 windows but predicts on 96; attention changes with set size | Accepted: added a 48-window mixer run; the 32-window run is a diagnostic |
| Query head as its own all-data family, blended per finding | Partly done (Y13 v2+v3 = 0.948); query-head all-data seeds are the night fallback |
| "Finish v3/320" | Misread (v3 = query head + view quota) |
| Gold-58 is overused; keep a held-out half | Accepted: decisions on the dev half; the test half is checked once at the end |
| Skip more seeds, 30 epochs, 448 px, CoAtNet, OA relabel | Agreed |

## 2026-10-13 23:40: Y14 = 0.946 like-for-like (v3 4 seeds vs Y8 0.948). Change the night backfill?
| Codex point | Our decision |
|---|---|
| Drop the query-head backfill: Y14 is a clean negative; Y13's 0.948 was diversity, not the query head | Dropped |
| Use the slots for held-out folds 2–3 of the mixer, no all-data until replicated | Queued v2 folds 2–3 (baseline) + mixer k48 folds 2–3 |
| Thresholds too low for 29-study halves: macro ≥0.01 OOF gain with the same direction across folds/halves; synovitis ≥0.05 on dev and confirmed on the other half | Adopted; no automatic all-data scaling tonight |

## 2026-10-14 05:40: why did the mixer collapse?
- Codex: "optimization shock". The zero gate first blocks the mixer's gradients, then opens at 10× the backbone lr and overwrites good features. Minimal fix: mixer and gate in the 1e-4 group; change nothing else. One retest only; drop the mixer if it's not within about 0.01–0.02 of the baseline. **Done (wmix2lr).**

## 2026-10-14 09:20: radiologist's side finding; best next test?
| Codex point | Our decision |
|---|---|
| Canonicalise sides (left → right layout), no mirror averaging; test on the unchanged v2 recipe, folds 0–1, no re-preprocessing | Done: wside running |
| Unknown sides stay as stored; audit the 2.5% where the tag and the geometry disagree | Unknown left as stored; audit later |
| Next: k 16 → 32 | Misread (already k32); skipped |
| Skip crop / slice-range / fat-sat changes until this is measured | Agreed |
| If it wins by ≥0.01: patch inference side handling, train all-data seeds, submit | Plan |

## 2026-10-14 11:40: DINOv2 ViT-S window family setup
| Codex point | Our decision |
|---|---|
| Backbone lr 2e-5, head 1e-3 is right | Used |
| Layer-wise lr decay 0.75–0.8 instead of freezing | Not in the first run: 2e-5 with warmup is a standard fine-tune; add decay only if it's unstable |
| Default pooling (CLS token) and 384 → 392 padding are fine; keep training and inference identical | Same WindowNet code path in both |

## 2026-10-14 15:00: how to make the radiologist better and not sycophantic
| Codex point | Our decision |
|---|---|
| Priorities: full resolution, blind dev-half reads, a rigid per-case form, running notes | Adopted |
| "Slices the reader relied on" only after the blind read, or it anchors | Adopted (step 2 of the protocol) |
| Blind-first protocol: confidence 1–5, supporting and contradicting slices, an alternative, a falsifier; then reveal; one primary cause | In the agent definition |
| A cocky persona can hallucinate decisive findings: require an "indeterminate" option, confidence, and calibration against blinded gold | Persona: "decisive where the images support it, indeterminate otherwise, never invent" |

## 2026-10-14 18:00: next 2 GPU jobs after the Y15 seeds
| Codex point | Our decision |
|---|---|
| (a) ConvNeXt side+v7 folds 0–1 first: gold/OOF for blend weights and the radiologist error review; validates Y15 | Queued (wside7) |
| (c) + synovitis 0.25 on the same folds: the only measured small lift; check it survives the stronger base | Queued (wside7syn) |
| Not CoAtNet side+v7 now: earlier CoAtNet blend gave no LB gain | Agreed for now; revisit if (a)/(c) show complementary errors |

## 2026-10-14 23:00: Y15 0.949 vs expected; why, and what next
| Codex point | Our decision |
|---|---|
| Ranked causes: selection noise, LB noise/rounding, side × v7 interaction, test-side fault, v7 test mismatch | The interaction is confirmed (side+v7 = v7 on gold) |
| Check test-time side detection and run an inference parity audit on labelled train DICOMs | Visible run: sides 2R/1L, 0 unknown; parity audit launched |
| Replicate 2×2 (base, side, v7, side+v7) on the same folds | Folds 2–3 for side+v7 and v7 queued |
| Locked protocol: untouched half, same-sign replication, blend only complementary models; keep Y8 as the anchor | Adopted |

## 2026-10-16: X5 labels (X2 recipe, stronger window-model teacher)
- **Verdict:** plausible but low expected value (~0 to +0.002 in a blend). The teacher is the same ConvNeXt/window family, so it mostly reinforces existing errors.
- **Flaw:** fold 4 (868 studies) has no out-of-fold teacher. Mixing X2 targets there with X5 elsewhere makes the label noise uneven. Either produce fold-4 OOF or exclude fold 4; don't mix.
- **Also:** check teacher calibration (per-finding mean, variance, extremes vs X2), since BCE cares about magnitude.
- **If run:** one all-data seed as a blend candidate, not a replacement. No new mix hyperparameters.
