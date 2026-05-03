@echo off
setlocal
cd /d %~dp0

echo [Compare architecture/checkpoints] LiteRaceSegNet vs SegFormer-B3
python seg\compare\compare_models.py ^
 --configs seg\config\pothole_binary_literace_train.yaml seg\config\pothole_binary_segformer_b3_train.yaml ^
 --names LiteRaceSegNet_CNN SegFormer_B3_Transformer ^
 --ckpts seg\runs\literace_boundary_degradation\best.pth seg\transformer_b3\checkpoints\segformer_b3_best.pth ^
 --input_dir datasets\pothole_binary\processed\val\images ^
 --mask_dir datasets\pothole_binary\processed\val\masks ^
 --device cpu ^
 --batch_size 1 ^
 --latency_warmup 10 ^
 --latency_repeats 50 ^
 --outdir seg\runs\model_compare
pause
