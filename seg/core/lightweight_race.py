from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
  def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, dilation=1, act=True):
    if p is None:
      p = dilation * (k // 2)
    layers = [
      nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=dilation, groups=groups, bias=False),
      nn.BatchNorm2d(out_ch),
    ]
    if act:
      layers.append(nn.Hardswish(inplace=True))
    super().__init__(*layers)


class DSConv(nn.Module):
  """Depthwise separable convolution used for lightweight feature extraction."""
  def __init__(self, in_ch, out_ch, k=3, s=1, dilation=1):
    super().__init__()
    p = dilation * (k // 2)
    self.net = nn.Sequential(
      ConvBNAct(in_ch, in_ch, k=k, s=s, p=p, groups=in_ch, dilation=dilation),
      ConvBNAct(in_ch, out_ch, k=1, s=1, p=0),
    )

  def forward(self, x):
    return self.net(x)


class LiteIRBlock(nn.Module):
  """MobileNetV2/V3-style lightweight block for the proposed road-damage model.

  This is not a direct copy of MobileNetV2/V3. It keeps the inverted-residual
  design principle but uses a simplified channel/layout choice for this thesis.
  """
  def __init__(self, in_ch, out_ch, stride=1, expand=2):
    super().__init__()
    hidden = max(in_ch, int(in_ch * expand))
    self.use_res = stride == 1 and in_ch == out_ch
    self.net = nn.Sequential(
      ConvBNAct(in_ch, hidden, k=1, s=1, p=0),
      ConvBNAct(hidden, hidden, k=3, s=stride, groups=hidden),
      nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
      nn.BatchNorm2d(out_ch),
    )
    self.act = nn.Hardswish(inplace=True)

  def forward(self, x):
    y = self.net(x)
    if self.use_res:
      y = y + x
    return self.act(y)


class LiteASPP(nn.Module):
  """Lightweight ASPP-like context module.

  The default rates=(1,2,4) are intentionally smaller than common full-ASPP
  settings so that the module stays aligned with the road-damage lightweight
  deployment scenario. Larger rates can be used for ablation.
  """
  def __init__(self, in_ch, out_ch, rates=(1, 2, 4)):
    super().__init__()
    rates = tuple(int(r) for r in rates)
    branch_ch = max(8, out_ch // 4)
    self.rates = rates
    self.branches = nn.ModuleList([
      DSConv(in_ch, branch_ch, k=3, s=1, dilation=r) for r in rates
    ])
    # Do not use BatchNorm after global 1x1 pooling; it can fail with
    # small batch sizes during training/profiling.
    self.pool = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=True),
      nn.Hardswish(inplace=True),
    )
    self.project = ConvBNAct(branch_ch * (len(rates) + 1), out_ch, k=1, s=1, p=0)

  def forward(self, x):
    h, w = x.shape[-2:]
    pooled = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
    return self.project(torch.cat([pooled] + [b(x) for b in self.branches], dim=1))


class BoundaryGate(nn.Module):
  """Predicts a boundary logit and optionally gates high-resolution detail features."""
  def __init__(self, detail_ch, context_ch, hidden):
    super().__init__()
    self.detail_proj = ConvBNAct(detail_ch, hidden, k=1, s=1, p=0)
    self.context_proj = ConvBNAct(context_ch, hidden, k=1, s=1, p=0)
    self.edge = nn.Sequential(
      DSConv(hidden * 2, hidden, k=3),
      nn.Conv2d(hidden, 1, kernel_size=1),
    )

  def forward(self, detail, context, apply_gate=True):
    size = detail.shape[-2:]
    d = self.detail_proj(detail)
    c = F.interpolate(self.context_proj(context), size=size, mode="bilinear", align_corners=False)
    edge = self.edge(torch.cat([d, c], dim=1))
    if apply_gate:
      gated_detail = detail * (1.0 + torch.sigmoid(edge))
    else:
      gated_detail = detail
    return gated_detail, edge


class LiteRaceSegNet(nn.Module):
  """Boundary-degradation-aware lightweight road-damage segmentation model.

  Contribution boundary:
  - Uses public ideas such as depthwise separable convolution, inverted-residual
   style blocks, and lightweight ASPP-style context aggregation.
  - The thesis contribution is the road-damage-specific detail/context/boundary
   wiring and the ablation-ready boundary-guided fusion protocol.
  """
  def __init__(
    self,
    num_classes,
    base_channels=24,
    context_channels=96,
    use_aux=True,
    use_detail_branch=True,
    context_module="lite_aspp",
    liteaspp_rates=(1, 2, 4),
    use_boundary_gate=True,
    fuse_boundary_logit=True,
  ):
    super().__init__()
    b = int(base_channels)
    c = int(context_channels)
    self.use_aux = bool(use_aux)
    self.use_detail_branch = bool(use_detail_branch)
    self.use_boundary_gate = bool(use_boundary_gate)
    self.fuse_boundary_logit = bool(fuse_boundary_logit)

    self.stem = nn.Sequential(
      ConvBNAct(3, b, k=3, s=2),
      DSConv(b, b, k=3, s=1),
    )

    self.detail = nn.Sequential(
      LiteIRBlock(b, b, stride=1, expand=2),
      LiteIRBlock(b, b, stride=1, expand=2),
    )

    self.down1 = nn.Sequential(
      LiteIRBlock(b, b * 2, stride=2, expand=2),
      LiteIRBlock(b * 2, b * 2, stride=1, expand=2),
    )
    self.down2 = nn.Sequential(
      LiteIRBlock(b * 2, b * 4, stride=2, expand=2),
      LiteIRBlock(b * 4, b * 4, stride=1, expand=2),
    )

    context_module = str(context_module).lower()
    if context_module in ("lite_aspp", "liteaspp", "aspp"):
      self.context = LiteASPP(b * 4, c, rates=liteaspp_rates)
    elif context_module in ("dsconv", "conv", "none"):
      self.context = nn.Sequential(
        DSConv(b * 4, c, k=3, s=1),
        DSConv(c, c, k=3, s=1),
      )
    else:
      raise ValueError(f"Unknown context_module: {context_module}")

    self.gate = BoundaryGate(b, c, hidden=max(16, b // 2))

    # The boundary logit is fused as an explicit channel by default to match
    # the thesis diagram. For ablation, it can be replaced with zeros.
    fuse_in = b + c + 1
    self.fuse = nn.Sequential(
      ConvBNAct(fuse_in, c, k=1, s=1, p=0),
      DSConv(c, c, k=3, s=1),
      nn.Dropout2d(0.05),
    )
    self.head = nn.Conv2d(c, num_classes, kernel_size=1)
    self.aux_head = nn.Conv2d(b * 2, num_classes, kernel_size=1) if self.use_aux else None

  def forward(self, x):
    size = x.shape[-2:]
    x2 = self.stem(x)

    if self.use_detail_branch:
      detail = self.detail(x2)
      down_source = detail
    else:
      # Ablation: remove the high-resolution detail contribution from fusion,
      # while keeping the context backbone valid by downsampling from x2.
      detail = torch.zeros_like(x2)
      down_source = x2

    x4 = self.down1(down_source)
    x8 = self.down2(x4)
    ctx = self.context(x8)

    gated_detail, edge_half = self.gate(detail, ctx, apply_gate=self.use_boundary_gate)
    ctx_up = F.interpolate(ctx, size=gated_detail.shape[-2:], mode="bilinear", align_corners=False)
    edge_for_fusion = edge_half if self.fuse_boundary_logit else torch.zeros_like(edge_half)
    fused = self.fuse(torch.cat([gated_detail, ctx_up, edge_for_fusion], dim=1))

    out = self.head(fused)
    out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
    boundary = F.interpolate(edge_half, size=size, mode="bilinear", align_corners=False)

    aux = None
    if self.aux_head is not None:
      aux = self.aux_head(x4)
      aux = F.interpolate(aux, size=size, mode="bilinear", align_corners=False)

    return OrderedDict(out=out, aux=aux, boundary=boundary)


def count_trainable_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
