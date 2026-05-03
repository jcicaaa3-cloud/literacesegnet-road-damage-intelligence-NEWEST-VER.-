"""SegFormer-B3 adapter for the capstone comparison pipeline.

Role in the project:
- This is a Transformer-family baseline candidate, not a replacement for the
 CNN/lightweight branch.
- It returns the same output dictionary shape as the existing CNN models:
 OrderedDict(out=..., aux=None, boundary=None)
- Real accuracy comparison requires a trained/fine-tuned checkpoint.
"""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_B3_CONFIG = {
  "num_channels": 3,
  "num_encoder_blocks": 4,
  "depths": [3, 4, 18, 3],
  "sr_ratios": [8, 4, 2, 1],
  "hidden_sizes": [64, 128, 320, 512],
  "patch_sizes": [7, 3, 3, 3],
  "strides": [4, 2, 2, 2],
  "num_attention_heads": [1, 2, 5, 8],
  "mlp_ratios": [4, 4, 4, 4],
  "decoder_hidden_size": 768,
  "hidden_dropout_prob": 0.0,
  "attention_probs_dropout_prob": 0.0,
  "classifier_dropout_prob": 0.1,
}

_VARIANT_CONFIGS = {
  "b3": _B3_CONFIG,
  "segformer-b3": _B3_CONFIG,
  "segformer_b3": _B3_CONFIG,
}


class SegFormerB3(nn.Module):
  """HuggingFace SegFormer wrapper for binary road-damage segmentation.

  By default this builds a B3-sized SegFormer from config so the code can be
  reviewed without downloading weights. Set pretrained=True in the YAML only
  when the environment can access/load the HuggingFace checkpoint.
  """

  def __init__(
    self,
    num_classes: int = 2,
    variant: str = "b3",
    pretrained: bool = False,
    hf_model_name: str = "nvidia/segformer-b3-finetuned-ade-512-512",
  ):
    super().__init__()
    try:
      from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except Exception as exc:
      raise RuntimeError(
        "SegFormer-B3를 사용하려면 transformers 패키지가 필요합니다. "
        "먼저 01_INSTALL_TRANSFORMER_OPTIONAL.bat 또는 "
        "pip install -r requirements_transformer_optional.txt 를 실행하세요."
      ) from exc

    self.num_classes = int(num_classes)
    self.variant = str(variant).lower()
    self.pretrained = bool(pretrained)
    self.hf_model_name = self._resolve_hf_model_name(hf_model_name)

    if self.variant not in _VARIANT_CONFIGS:
      raise ValueError(f"Unsupported SegFormer-B3 variant: {variant}")

    if self.pretrained:
      self.model = SegformerForSemanticSegmentation.from_pretrained(
        self.hf_model_name,
        num_labels=self.num_classes,
        ignore_mismatched_sizes=True,
      )
    else:
      cfg_dict = dict(_VARIANT_CONFIGS[self.variant])
      cfg_dict.update({
        "num_labels": self.num_classes,
        "id2label": {i: f"class_{i}" for i in range(self.num_classes)},
        "label2id": {f"class_{i}": i for i in range(self.num_classes)},
      })
      config = SegformerConfig(**cfg_dict)
      self.model = SegformerForSemanticSegmentation(config)


  @staticmethod
  def _resolve_hf_model_name(hf_model_name: str) -> str:
    """Resolve a local relative HF folder while still allowing Hub model IDs."""
    name = str(hf_model_name)
    # Hub IDs such as nvidia/segformer-b3-finetuned-ade-512-512 are not
    # local folders. Only rewrite the path when it exists inside the project.
    p = Path(name)
    if p.exists():
      return str(p.resolve())
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / p
    if candidate.exists():
      return str(candidate.resolve())
    return name

  def forward(self, x: torch.Tensor):
    size = x.shape[-2:]
    y = self.model(pixel_values=x)
    logits = y.logits
    logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return OrderedDict(out=logits, aux=None, boundary=None)
