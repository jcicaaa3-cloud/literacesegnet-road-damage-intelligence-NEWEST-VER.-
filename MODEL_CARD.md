# Model card: LiteRaceSegNet

## Model name

LiteRaceSegNet

## Task

Semantic segmentation for road-surface damage such as potholes, cracks, and local surface damage. The default configuration is binary segmentation:

- background: `0`
- road damage: `1`

## Intended use

- capstone project evaluation
- computer vision portfolio review
- local experimentation on permitted road-damage image-mask datasets
- CPU/GPU latency and model-size comparison against a Transformer baseline

## Not intended for

- safety-critical road maintenance decisions without human review
- production deployment without dataset validation, failure analysis, and field testing
- use on datasets whose license does not allow training or redistribution
- claiming universal superiority over all segmentation models

## Architecture summary

LiteRaceSegNet uses a lightweight CNN design with:

- detail branch for local boundary and thin-structure information
- context branch with LiteASPP-style aggregation
- boundary auxiliary output
- boundary-guided fusion
- segmentation head for final mask prediction

## Baseline comparison

SegFormer-B3 is kept as a separate Transformer baseline. It is not merged into LiteRaceSegNet and is not presented as the proposed model.

The intended comparison includes:

- mIoU
- Damage IoU
- Pixel Accuracy
- parameter count
- estimated FP32 parameter size
- CPU latency
- CUDA latency
- throughput FPS
- CUDA memory usage

## Evaluation protocol

CPU and GPU results should be interpreted separately.

- CPU result: no-GPU / field deployment evidence
- GPU result: AWS / CUDA acceleration evidence

Do not compare CPU latency and GPU latency as if they were the same condition. Compare LiteRaceSegNet vs SegFormer-B3 within the same device condition.

## Limitations

- Small and irregular road damage can be missed under shadows, lane markings, water stains, and low contrast.
- Metrics depend heavily on dataset quality and annotation consistency.
- Boundary quality should be reviewed visually, not only through mIoU.
- Latency depends on hardware, image size, thread count, batch size, CUDA driver state, and background load.

## License note

LiteRaceSegNet source code in this repository is under the MIT License. Datasets, pretrained weights, Hugging Face files, and third-party libraries are not relicensed by this repository.