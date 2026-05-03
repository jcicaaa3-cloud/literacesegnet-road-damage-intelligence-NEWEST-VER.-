# Result table template

실험 후 아래 표를 README나 보고서에 옮긴다. 수치는 만들지 말고 `final_evidence` 결과에서 가져온다.

## CPU condition

| Model | Params | FP32 size | CPU latency mean | CPU latency std | FPS | mIoU | Damage IoU | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LiteRaceSegNet | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] | Proposed lightweight CNN |
| SegFormer-B3 | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] | Transformer baseline |

## GPU condition

| Model | CUDA latency mean | CUDA latency std | Throughput FPS | CUDA peak memory | AMP | mIoU | Damage IoU | Note |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| LiteRaceSegNet | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] | Proposed lightweight CNN |
| SegFormer-B3 | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] | Transformer baseline |

## Safe interpretation sentence

> Under the same validation layout and the same device condition, LiteRaceSegNet shows [lower/higher/similar] latency and [lower/higher/similar] model cost compared with SegFormer-B3 while achieving [fill] Damage IoU. This supports the interpretation that LiteRaceSegNet provides [fill] trade-off for lightweight road-damage segmentation.