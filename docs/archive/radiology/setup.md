# What our automated reader sees (plain clinical description)

## The task
Each knee MRI study must be graded for 12 findings, each present or absent. The reference standard is MSK radiologists reading the **images**, using strict thresholds. Borderline or "on the fence" findings were graded **negative**.

| Finding | Positive only if |
|---|---|
| ACL tear | high-grade partial (>50% of fibres) or complete; sprain, mucoid degeneration or an intact graft = negative |
| MCL tear | high-grade partial or complete acute tear; low-grade sprain or chronic change = negative |
| Medial / lateral meniscus tear | signal reaching the articular surface, or a displaced, truncated, flap, bucket-handle or root tear; intrasubstance degenerative signal = negative |
| Medial / lateral / patellofemoral OA | high-grade cartilage loss (>50% thickness, e.g. full-thickness, grade 3–4) over a moderate or large area (about ≥1 cm) in that compartment; mild or moderate chondropathy, fissuring or osteophytes alone = negative |
| Effusion | moderate or large; small, trace or physiological = negative |
| Synovitis | synovial thickening, proliferation or inflammation (including Hoffa synovitis) |
| Baker's cyst | moderate or large |
| Bone contusion | traumatic bone-marrow oedema |
| Acute fracture | acute cortical break or fracture line |

## How the images reach the reader
- **Series:** from each study, one series per slot:
  - Sagittal fluid-sensitive fat-suppressed (sag FS)
  - Coronal FS
  - Axial FS
  - Sagittal non-FS (usually PD or T1)
  - Coronal non-FS

  "FS" means fluid bright and fat suppressed. A missing series leaves an empty slot.
- **Slices (current version, after consultation 1):** the outer 3% of slices at each end of the stack are dropped. From the central 94%, up to this many **evenly spaced** slices are kept: sag FS 26, cor FS 30, axial FS 30, sag non-FS 22, cor non-FS 20. Short series keep all their slices, without repeats. Earlier version: 6% trim and 18 / 18 / 12 for cor FS / axial / cor non-FS.
- **Side:** left knees are now mirrored into the right-knee layout before reading. The case images in `cases2/` are shown as stored (not mirrored); `cases2/knees.csv` gives each knee's side.
- **In-plane:** a 140 mm × 140 mm square centred on the body outline, resized to 384 × 384 px (about 0.36 mm per pixel). Intensity is normalised per series.
- **How the reader judges:** it looks at small stacks of 3 neighbouring slices, one stack at a time. It knows which series and roughly which position in the stack each one comes from, and it combines evidence from all stacks for each finding. It does not explicitly match a location on one plane to the same location on another plane.
- **What it learned from:** mainly the free-text radiology reports of about 4,400 studies (not image labels). So whatever reports routinely omit or describe loosely, the reader also learns loosely.

## Where it is weakest (agreement with the radiologists' image reads)
- Strong (0.94–0.99): ACL, MCL, medial meniscus, medial OA, effusion, Baker's cyst, contusion, fracture.
- Weak:

| Finding | Agreement |
|---|---|
| Synovitis | 0.82–0.85 |
| Lateral OA | 0.84 |
| PF OA | 0.88 |
| Lateral meniscus | 0.87–0.90 |

- **Synovitis:** about 85% of reports never mention it, yet about a third of those knees are synovitis-positive on the radiologists' image read.

## Case files (`cases/`)
- **Images:** `caseNN_<slot>.png` shows the exact slices the reader receives for that series, numbered in stack order, 6 per row.
- **Case list:** `case_list.csv` says, for each case, which weak finding it illustrates: missed (radiologists positive, reader confidently negative), false alarm (radiologists negative, reader positive) or caught.
- **Radiologist read:** `radiologist_labels.csv` gives the radiologists' read for the relevant findings.
