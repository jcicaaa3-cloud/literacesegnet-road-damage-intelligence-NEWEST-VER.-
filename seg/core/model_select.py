from core.network import MicroSegNet, TinyUNet
from core.lightweight_race import LiteRaceSegNet


def get_model(cfg):
  model_cfg = cfg['model']
  name = model_cfg['name'].lower()
  num_classes = int(model_cfg['num_classes'])
  use_aux = bool(model_cfg.get('use_aux', True))

  if name == 'lite_race':
    return LiteRaceSegNet(
      num_classes=num_classes,
      base_channels=int(model_cfg.get('base_channels', 24)),
      context_channels=int(model_cfg.get('context_channels', 96)),
      use_aux=use_aux,
      use_detail_branch=bool(model_cfg.get('use_detail_branch', True)),
      context_module=model_cfg.get('context_module', 'lite_aspp'),
      liteaspp_rates=tuple(model_cfg.get('liteaspp_rates', [1, 2, 4])),
      use_boundary_gate=bool(model_cfg.get('use_boundary_gate', True)),
      fuse_boundary_logit=bool(model_cfg.get('fuse_boundary_logit', True)),
    )

  if name in ('segformer_b3', 'segformerb3', 'segformer-b3', 'segformer_b3'):
    try:
      from transformer_b3.segformer_b3_adapter import SegFormerB3
    except Exception as exc:
      raise RuntimeError(
        'SegFormer-B3 모델을 불러오지 못했습니다. '
        'transformers 설치가 필요하면 01_INSTALL_TRANSFORMER_OPTIONAL.bat를 실행하세요.'
      ) from exc
    return SegFormerB3(
      num_classes=num_classes,
      variant=model_cfg.get('variant', 'b3'),
      pretrained=bool(model_cfg.get('pretrained', False)),
      hf_model_name=model_cfg.get('hf_model_name', 'nvidia/segformer-b3-finetuned-ade-512-512'),
    )

  if name == 'tiny_unet':
    return TinyUNet(
      num_classes=num_classes,
      base_channels=int(model_cfg.get('base_channels', 16)),
      use_aux=use_aux,
    )

  if name == 'micro_seg':
    return MicroSegNet(
      num_classes=num_classes,
      base_channels=int(model_cfg.get('base_channels', 8)),
      use_aux=use_aux,
    )

  raise ValueError(f"Unsupported service model name: {name}")
