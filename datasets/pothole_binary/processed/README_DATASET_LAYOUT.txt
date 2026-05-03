Pothole / road-damage binary segmentation dataset layout
=======================================================

This repository does not include dataset images or masks.
Place only permitted image-mask pairs here before training.

Required layout:

 datasets/pothole_binary/processed/
  train/
   images/
    road_001.jpg
    road_002.jpg
   masks/
    road_001.png
    road_002.png
  val/
   images/
    road_101.jpg
   masks/
    road_101.png

Mask rule:
- background = 0 / black
- pothole or damage region = any value greater than 0, commonly 255

Filename rule:
- image and mask should share the same stem.
 Example: road_001.jpg + road_001.png
- Also accepted: road_001_mask.png, road_001_gt.png, road_001_label.png

Important:
- Fine-tuning cannot be done with road images alone.
- It requires pixel-level mask labels.
- Do not commit raw images, masks, paid datasets, or private road footage.
- Follow the dataset source license and redistribution terms.

Pairing checker:

 python seg/tools/check_dataset_pairs.py --root datasets/pothole_binary/processed

Pairing reports are saved to:

 seg/runs/dataset_pairing_reports/

If a file is ambiguous or unmatched, training stops so the wrong mask is not silently used.
