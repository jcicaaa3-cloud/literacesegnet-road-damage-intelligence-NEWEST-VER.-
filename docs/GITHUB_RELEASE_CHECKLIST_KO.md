# GitHub 공개 전 체크리스트

취업용 공개 저장소로 올리기 전에 이 목록을 확인한다.

## 1. 파일 포함 여부

올려도 됨:

- `seg/`, `llm_service/`, `scripts/`, `docs/`
- `README.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`
- `requirements.txt`, `requirements_service.txt`, `requirements_transformer_optional.txt`
- `.gitignore`
- dataset layout README와 `.gitkeep`

올리면 안 됨:

- 실제 dataset 이미지/mask
- checkpoint: `.pth`, `.pt`, `.ckpt`, `.safetensors`, `.bin`
- Hugging Face cache folder
- generated overlay images, CSV, JSON, logs
- `.env`, API key, AWS key
- 논문 DOCX/PDF 원본
- 개인정보가 들어간 파일

## 2. README 주장 톤

좋음:

> LiteRaceSegNet is evaluated as a lightweight CNN candidate against SegFormer-B3 under CPU and GPU conditions.

위험함:

> LiteRaceSegNet is always better than SegFormer.

결과표가 나오기 전에는 수치를 단정하지 않는다.

## 3. License 문구

README에 반드시 남길 것:

> The source code written for this repository is MIT-licensed. External datasets, pretrained weights, model files, and APIs are not included and are not relicensed.

## 4. 면접관이 볼 포인트

README 첫 화면에서 바로 보여야 하는 것:

- 직접 만든 모델이 뭔지
- baseline이 뭔지
- dataset/weight를 왜 안 넣었는지
- CPU/GPU 비교를 왜 했는지
- 결과 파일이 어디 생기는지

## 5. 마지막 검사 명령

```bash
git status --short
find . -type f \( -name '*.pth' -o -name '*.pt' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.bin' -o -name '*.env' -o -name '*.docx' -o -name '*.pdf' \)
```

위 명령에서 민감한 파일이 나오면 commit 전에 제거한다.