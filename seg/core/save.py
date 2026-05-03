import torch


def save_state(path, model, optimizer, scheduler, scaler, epoch, best, cfg):
  state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict() if optimizer is not None else None,
    'scheduler': scheduler.state_dict() if scheduler is not None else None,
    'scaler': scaler.state_dict() if scaler is not None else None,
    'epoch': epoch,
    'best': best,
    'config': cfg,
  }
  torch.save(state, path)


def load_state(path, model, optimizer=None, scheduler=None, scaler=None, map_location='cpu'):
  ckpt = torch.load(path, map_location=map_location)
  model.load_state_dict(ckpt['model'], strict=True)
  if optimizer is not None and ckpt.get('optimizer') is not None:
    optimizer.load_state_dict(ckpt['optimizer'])
  if scheduler is not None and ckpt.get('scheduler') is not None:
    scheduler.load_state_dict(ckpt['scheduler'])
  if scaler is not None and ckpt.get('scaler') is not None:
    scaler.load_state_dict(ckpt['scaler'])
  return ckpt
