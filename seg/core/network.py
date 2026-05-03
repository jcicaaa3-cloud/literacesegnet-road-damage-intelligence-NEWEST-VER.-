from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Sequential):
  def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1, act=nn.ReLU):
    super().__init__(
      nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
      nn.BatchNorm2d(out_ch),
      act(inplace=True),
    )


class DWConv(nn.Module):
  def __init__(self, in_ch, out_ch, k=3, s=1, d=1):
    super().__init__()
    p = d * (k // 2)
    self.block = nn.Sequential(
      ConvBnAct(in_ch, in_ch, k=k, s=s, p=p, groups=in_ch),
      ConvBnAct(in_ch, out_ch, k=1, s=1, p=0, groups=1),
    )

  def forward(self, x):
    return self.block(x)


class ContextBlock(nn.Module):
  def __init__(self, in_ch, out_ch=128, rates=(1, 3, 6)):
    super().__init__()
    branch_ch = out_ch // 4
    self.pool = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      ConvBnAct(in_ch, branch_ch, k=1, s=1, p=0),
    )
    self.branches = nn.ModuleList([DWConv(in_ch, branch_ch, k=3, d=r) for r in rates])
    self.out = ConvBnAct(branch_ch * (len(rates) + 1), out_ch, k=1, s=1, p=0)

  def forward(self, x):
    h, w = x.shape[-2:]
    pooled = self.pool(x)
    pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
    parts = [pooled] + [m(x) for m in self.branches]
    return self.out(torch.cat(parts, dim=1))


class EdgeHead(nn.Module):
  def __init__(self, low_ch, mid_ch, ctx_ch, hidden=64):
    super().__init__()
    self.low = ConvBnAct(low_ch, hidden, k=1, s=1, p=0)
    self.mid = ConvBnAct(mid_ch, hidden, k=1, s=1, p=0)
    self.ctx = ConvBnAct(ctx_ch, hidden, k=1, s=1, p=0)
    self.out = nn.Sequential(
      DWConv(hidden * 3, hidden, k=3),
      nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
    )

  def forward(self, low, mid, ctx):
    size = low.shape[-2:]
    low = self.low(low)
    mid = F.interpolate(self.mid(mid), size=size, mode='bilinear', align_corners=False)
    ctx = F.interpolate(self.ctx(ctx), size=size, mode='bilinear', align_corners=False)
    return self.out(torch.cat([low, mid, ctx], dim=1))


class MergeBlock(nn.Module):
  def __init__(self, low_ch, mid_ch, ctx_ch, out_ch=128):
    super().__init__()
    self.low = ConvBnAct(low_ch, 48, k=1, s=1, p=0)
    self.mid = ConvBnAct(mid_ch, 48, k=1, s=1, p=0)
    self.ctx = ConvBnAct(ctx_ch, 64, k=1, s=1, p=0)
    self.out = nn.Sequential(
      DWConv(48 + 48 + 64, out_ch, k=3),
      nn.Dropout2d(0.1),
    )

  def forward(self, low, mid, ctx, edge_logit):
    size = low.shape[-2:]
    gate = torch.sigmoid(edge_logit)
    low = self.low(low) * (1.0 + gate)
    mid = F.interpolate(self.mid(mid), size=size, mode='bilinear', align_corners=False)
    ctx = F.interpolate(self.ctx(ctx), size=size, mode='bilinear', align_corners=False)
    return self.out(torch.cat([low, mid, ctx], dim=1))


class SegHead(nn.Sequential):
  def __init__(self, in_ch, num_classes):
    super().__init__(
      DWConv(in_ch, in_ch, k=3),
      nn.Dropout2d(0.1),
      nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True),
    )


class Mbv3EdgeNet(nn.Module):
  def __init__(self, num_classes, backbone_weights='imagenet', use_aux=True):
    super().__init__()
    try:
      from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
      from torchvision.models._utils import IntermediateLayerGetter
    except Exception as exc:
      raise RuntimeError(
        'Mbv3EdgeNet 생성 실패: torchvision import 오류가 있습니다. '
        'tiny_unet 설정으로 먼저 실행하거나 torch/torchvision 버전을 맞춰주세요.'
      ) from exc

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if backbone_weights == 'imagenet' else None
    body = mobilenet_v3_large(weights=weights, dilated=True).features

    stage_idx = [0] + [i for i, b in enumerate(body) if getattr(b, '_is_cn', False)] + [len(body) - 1]
    low_idx = stage_idx[-4]
    mid_idx = stage_idx[-2]
    high_idx = stage_idx[-1]

    self.body = IntermediateLayerGetter(body, return_layers={str(low_idx): 'low', str(mid_idx): 'mid', str(high_idx): 'high'})

    low_ch = body[low_idx].out_channels
    mid_ch = body[mid_idx].out_channels
    high_ch = body[high_idx].out_channels

    self.ctx = ContextBlock(high_ch, 128)
    self.edge = EdgeHead(low_ch, mid_ch, 128, 64)
    self.merge = MergeBlock(low_ch, mid_ch, 128, 128)
    self.cls = SegHead(128, num_classes)
    self.use_aux = use_aux
    self.aux = SegHead(mid_ch, num_classes) if use_aux else None

  def forward(self, x):
    size = x.shape[-2:]
    feat = self.body(x)
    low = feat['low']
    mid = feat['mid']
    high = feat['high']

    ctx = self.ctx(high)
    edge = self.edge(low, mid, ctx)
    mix = self.merge(low, mid, ctx, edge)
    out = self.cls(mix)

    out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
    edge = F.interpolate(edge, size=size, mode='bilinear', align_corners=False)

    aux = None
    if self.aux is not None:
      aux = self.aux(mid)
      aux = F.interpolate(aux, size=size, mode='bilinear', align_corners=False)

    return OrderedDict(out=out, aux=aux, boundary=edge)


class DoubleConv(nn.Sequential):
  def __init__(self, in_ch, out_ch):
    super().__init__(
      nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
  def __init__(self, in_ch, skip_ch, out_ch):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.conv = DoubleConv(in_ch + skip_ch, out_ch)

  def forward(self, x, skip):
    x = self.up(x)
    if x.shape[-2:] != skip.shape[-2:]:
      x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
    x = torch.cat([x, skip], dim=1)
    return self.conv(x)


class MicroSegNet(nn.Module):
  def __init__(self, num_classes, base_channels=8, use_aux=True):
    super().__init__()
    b = int(base_channels)
    self.stem = nn.Sequential(
      ConvBnAct(3, b, k=3, s=2, p=1),
      ConvBnAct(b, b, k=3, s=1, p=1),
    )
    self.enc2 = nn.Sequential(
      ConvBnAct(b, b * 2, k=3, s=2, p=1),
      ConvBnAct(b * 2, b * 2, k=3, s=1, p=1),
    )
    self.enc3 = nn.Sequential(
      ConvBnAct(b * 2, b * 4, k=3, s=2, p=1),
      ConvBnAct(b * 4, b * 4, k=3, s=1, p=1),
    )
    self.fuse = ConvBnAct(b + b * 2 + b * 4, b * 2, k=1, s=1, p=0)
    self.head = nn.Conv2d(b * 2, num_classes, kernel_size=1)
    self.boundary = nn.Conv2d(b * 2, 1, kernel_size=1)
    self.use_aux = use_aux
    self.aux = nn.Conv2d(b * 4, num_classes, kernel_size=1) if use_aux else None

  def forward(self, x):
    size = x.shape[-2:]
    x1 = self.stem(x)
    x2 = self.enc2(x1)
    x3 = self.enc3(x2)
    x2_up = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
    x3_up = F.interpolate(x3, size=x1.shape[-2:], mode='bilinear', align_corners=False)
    fused = self.fuse(torch.cat([x1, x2_up, x3_up], dim=1))
    out = self.head(fused)
    boundary = self.boundary(fused)
    out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
    boundary = F.interpolate(boundary, size=size, mode='bilinear', align_corners=False)
    aux = None
    if self.aux is not None:
      aux = self.aux(x3)
      aux = F.interpolate(aux, size=size, mode='bilinear', align_corners=False)
    return OrderedDict(out=out, aux=aux, boundary=boundary)


class TinyUNet(nn.Module):
  def __init__(self, num_classes, base_channels=16, use_aux=True):
    super().__init__()
    b = int(base_channels)
    self.enc1 = DoubleConv(3, b)
    self.enc2 = DoubleConv(b, b * 2)
    self.enc3 = DoubleConv(b * 2, b * 4)
    self.bottleneck = DoubleConv(b * 4, b * 8)
    self.pool = nn.MaxPool2d(2)

    self.dec3 = UpBlock(b * 8, b * 4, b * 4)
    self.dec2 = UpBlock(b * 4, b * 2, b * 2)
    self.dec1 = UpBlock(b * 2, b, b)

    self.head = nn.Conv2d(b, num_classes, kernel_size=1)
    self.boundary = nn.Conv2d(b, 1, kernel_size=1)
    self.use_aux = use_aux
    self.aux = nn.Conv2d(b * 2, num_classes, kernel_size=1) if use_aux else None

  def forward(self, x):
    size = x.shape[-2:]
    x1 = self.enc1(x)
    x2 = self.enc2(self.pool(x1))
    x3 = self.enc3(self.pool(x2))
    x4 = self.bottleneck(self.pool(x3))

    d3 = self.dec3(x4, x3)
    d2 = self.dec2(d3, x2)
    d1 = self.dec1(d2, x1)

    out = self.head(d1)
    boundary = self.boundary(d1)
    out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
    boundary = F.interpolate(boundary, size=size, mode='bilinear', align_corners=False)

    aux = None
    if self.aux is not None:
      aux = self.aux(d2)
      aux = F.interpolate(aux, size=size, mode='bilinear', align_corners=False)

    return OrderedDict(out=out, aux=aux, boundary=boundary)
