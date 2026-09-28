understanding the problem:

an mri takes stack of thin 2d slices that form a 3d image.
series: one stack of slices, taken from one direction with one scanner settings. ~30 DICOM files.
study: one patient's full knee exam. comprises of multiple series.

for each knee study we need to predict probability for 12 yes/no questions. 
Group	                    Labels
Torn ligaments	            ACL, MCL
Torn cartilage pads	        Medial Meniscus, Lateral Meniscus
Worn-out joint (arthritis)	Medial OA, Lateral OA, PF OA
Fluid/swelling	            Effusion, Synovitis, Baker's cyst
Bone damage	                Contusion (bruise), Fracture

in this problem most studies have no labels. but every training study comes with a radiologists report.
at test time we only get images.
text - labels psuedo labelling

