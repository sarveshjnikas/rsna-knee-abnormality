# Radiologist running notes

## Consultation 1 (2026-10-14): 15 cases, tiles at 256 px; given labels and the reader's errors up front (not blind)
- **Side / orientation (suspected):** the sagittal stack direction flips with the knee side. Verified by the team on 6/6 cases against DICOM laterality. Canonicalising the side gave +0.004 on gold across 4 folds, with gains in lateral OA, PF OA and medial meniscus, and a drop in lateral meniscus (0.891 → 0.876).
- **Coverage:** only ~6/18 coronal and ~4–6/18 axial slices cross the joint; the 6% trim drops the proximal suprapatellar recess. → v7 (denser coronal/axial slices, 3% trim) is under test.
- **Crop:** the patella touches or is cut by the anterior edge in cases 10 and 11. Not addressed yet.
- **Synovitis:** read from Hoffa fat-pad signal and fluid heterogeneity / lining, not effusion volume. Not addressed directly.
- **Fat-suppression failure:** cases 06, 07, 13. Not addressed.
- **Limits of this consultation:** cartilage and meniscal surfaces were not assessable at 256 px, and the reads were not blind.

## Consultation 2 (2026-10-15): blind-first, 384 px, 9 dev-half knees, side+v7 reader errors
- **Blind reliability:** 11/19 decisive calls matched the labels; 17/36 were indeterminate. The radiologist is stricter than the labellers on synovitis.
- **The 12 reader errors split about evenly:**
  - **Fixable by viewing (4):** anterior-horn coverage (knee02), cross-plane patellofemoral (knee01), posterior condyle coverage (knee08), cruciate/recess synovitis (knee09).
  - **Reader wrong on visible findings (4):** knee04 LM, knee03 LM, knee07 LOA, knee02 PF. These need capacity or data, not a different view.
  - **Borderline / label doubt (4):** knee05 LOA, knee06 PF, knee05 synovitis, knee06 synovitis.
- **Corrections:** Hoffa signal *does* count toward synovitis under these labels (the phase-1 challenge was withdrawn). Pixel size is no longer the limit at 384 px; measuring area across planes is.
- **Coronal stack:** ~1/3 of the 30 slices are pre-patellar soft tissue or calf; restrict it to the condyle–plateau extent.
