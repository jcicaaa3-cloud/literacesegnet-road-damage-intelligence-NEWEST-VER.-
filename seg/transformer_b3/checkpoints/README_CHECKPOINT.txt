이 폴더에는 SegFormer-B3 fine-tuning 결과 checkpoint를 넣습니다.

권장 파일명 예시:
- segformer_b3_best.pth
- segformer_b3_epoch20.pth

현재 ZIP에는 실제 학습된 SegFormer-B3 checkpoint가 포함되어 있지 않습니다.
따라서 비교 스크립트는 checkpoint가 없을 경우 구조/파라미터/추론시간 중심으로만 비교하고, mIoU 같은 정확도 지표는 NA로 표시합니다.
