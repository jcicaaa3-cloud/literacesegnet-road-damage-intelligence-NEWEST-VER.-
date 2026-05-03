import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch

from core.model_select import get_model
from core.save import load_state
from core.train_utils import load_yaml, make_dir, get_device

IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')


def get_args():
  p = argparse.ArgumentParser(description='Batch inference for a trained segmentation checkpoint.')
  p.add_argument('--config', type=str, required=True)
  p.add_argument('--ckpt', type=str, required=True)
  p.add_argument('--input_dir', type=str, required=True)
  p.add_argument('--output_dir', type=str, required=True)
  return p.parse_args()


def collect_images(folder):
  items = []
  for ext in IMG_EXTS:
    items.extend(glob(os.path.join(folder, ext)))
    items.extend(glob(os.path.join(folder, ext.upper())))
  return sorted(set(items))


def prep(img, image_size):
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  rs = cv2.resize(rgb, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
  arr = rs.astype(np.float32) / 255.0
  mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
  arr = (arr - mean) / std
  arr = np.transpose(arr, (2, 0, 1))
  return torch.from_numpy(arr).unsqueeze(0).float(), rs


def color_mask(mask, palette):
  canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
  for idx, color in enumerate(palette):
    canvas[mask == idx] = np.array(color, dtype=np.uint8)
  return canvas


@torch.no_grad()
def main():
  args = get_args()
  cfg = load_yaml(args.config)
  device = get_device(cfg)
  make_dir(args.output_dir)

  net = get_model(cfg).to(device)
  load_state(args.ckpt, net, map_location=device.type)
  net.eval()

  image_list = collect_images(args.input_dir)
  if not image_list:
    raise FileNotFoundError(args.input_dir)

  palette = cfg['infer']['palette']
  alpha = float(cfg['infer'].get('overlay_alpha', 0.45))
  image_size = cfg['train']['image_size']

  for path in image_list:
    org = cv2.imread(path, cv2.IMREAD_COLOR)
    if org is None:
      continue
    x, resized = prep(org, image_size)
    x = x.to(device)
    out = net(x)
    pred = torch.argmax(out['out'], dim=1)[0].cpu().numpy().astype(np.uint8)
    mask = color_mask(pred, palette)
    overlay = cv2.addWeighted(
      cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
      1.0 - alpha,
      cv2.cvtColor(mask, cv2.COLOR_RGB2BGR),
      alpha,
      0,
    )
    name = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(args.output_dir, f'{name}_overlay.png'), overlay)
    cv2.imwrite(os.path.join(args.output_dir, f'{name}_mask_color.png'), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output_dir, f'{name}_pred_class.png'), pred.astype(np.uint8))

  print(f'[OK] Saved inference outputs to: {args.output_dir}')


if __name__ == '__main__':
  main()
