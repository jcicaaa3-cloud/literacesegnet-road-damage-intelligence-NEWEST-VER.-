SegFormer-B3 폴더 설명
======================

이 폴더는 CNN 계열 모델을 대체하기 위한 폴더가 아니라, Transformer 계열 비교 baseline을 분리해서 관리하기 위한 공간입니다.

발표/보고서에서의 정확한 설명:
- 기존 lite_race 계열: 경량 CNN + LiteASPP + boundary-guided fusion 후보
- SegFormer-B3: Transformer 계열 비교 후보
- 비교 목적: 어느 쪽이 무조건 우월한지 주장하는 것이 아니라, 도로 손상 segmentation에서 경계 표현, 작은 손상, 추론 비용, 모델 크기, 서비스 적용 가능성을 비교하는 것

현재 포함 파일:
- segformer_b3_adapter.py
 HuggingFace SegFormer를 기존 pipeline 출력 형식에 맞춘 adapter입니다.
 출력 형식은 기존 모델과 동일하게 OrderedDict(out=..., aux=None, boundary=None)을 반환합니다.

주의:
- transformers 패키지가 없으면 SegFormer-B3 모델은 생성되지 않습니다.
- 실제 성능 비교를 하려면 SegFormer-B3도 pothole/road damage 데이터셋으로 fine-tuning한 checkpoint가 필요합니다.
- pretrained=True는 인터넷 또는 캐시된 HuggingFace weight가 있을 때만 사용하세요.
