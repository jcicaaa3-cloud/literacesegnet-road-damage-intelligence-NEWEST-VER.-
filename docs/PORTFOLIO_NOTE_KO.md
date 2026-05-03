# 포트폴리오 설명 메모

## 1. 한 줄 소개

도로 손상 영역을 픽셀 단위로 분할하는 PyTorch segmentation 프로젝트입니다. 직접 설계한 LiteRaceSegNet과 SegFormer-B3 baseline을 분리해 학습하고, CPU/GPU 조건에서 모델 크기·latency·mask 품질을 비교합니다.

## 2. 취업용으로 보이는 포인트

- 직접 만든 모델 구조가 있다.
- baseline을 섞지 않고 분리했다.
- dataset pairing checker가 있다.
- 결과를 overlay 이미지뿐 아니라 CSV/JSON/markdown 표로 남긴다.
- CPU와 AWS GPU 조건을 나눠 latency와 throughput을 측정한다.
- dataset, checkpoint, pretrained weights, API key를 GitHub에 넣지 않는다.

## 3. 면접 답변 예시

Q. 왜 SegFormer를 넣었나?

A. 직접 만든 LiteRaceSegNet의 위치를 확인하기 위해서입니다. SegFormer-B3는 강한 Transformer baseline으로 두고, 제안 모델인 LiteRaceSegNet과 같은 validation layout에서 비교했습니다.

Q. LiteRaceSegNet의 차이는 무엇인가?

A. 도로 손상은 작고 경계가 불규칙한 경우가 많습니다. LiteRaceSegNet은 detail branch와 boundary branch를 둬서 downsampling 과정에서 약해지는 세부 위치 정보를 보완하려고 했습니다.

Q. CPU/GPU를 왜 둘 다 측정했나?

A. CPU는 GPU 없는 환경에서의 실사용 가능성을 보기 위한 조건이고, GPU는 AWS 가속 환경에서 throughput과 memory 사용량을 보기 위한 조건입니다.

Q. 공개 GitHub에 dataset과 checkpoint가 왜 없나?

A. dataset과 weight는 각자 라이선스가 있어서 임의로 재배포하면 안 됩니다. 이 저장소는 코드와 실행 구조를 공개하고, 사용자가 허용된 dataset을 넣어 재현하도록 구성했습니다.