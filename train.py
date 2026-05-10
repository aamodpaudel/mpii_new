from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MPIIGazeNormalizedDataset, split_subjects
from src.model import GazeRegressor
from src.model_gazecapture_ft import GazeCaptureEyeFTModel
from src.utils import angular_error_deg, ensure_dir, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/mpii_baseline.yaml')
    p.add_argument('--model_type', type=str, default='baseline', choices=['baseline', 'gazecapture_ft'])
    p.add_argument('--gazecapture_ckpt', type=str, default='')
    return p.parse_args()


def load_gazecapture_eye_weights(model: GazeCaptureEyeFTModel, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict']
    eye_state = {}
    for k, v in sd.items():
        if k.startswith('eyeModel.features.'):
            eye_state[k.replace('eyeModel.', '')] = v
    missing, unexpected = model.eyeModel.load_state_dict(eye_state, strict=False)
    print(f'loaded_gazecapture_eye_weights missing={len(missing)} unexpected={len(unexpected)}')


def set_eye_backbone_trainable(model, trainable: bool) -> None:
    if hasattr(model, 'eyeModel'):
        for p in model.eyeModel.parameters():
            p.requires_grad = trainable


def run_epoch(model, loader, optimizer, device, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.03)

    total_loss = 0.0
    total_ang = 0.0
    total_n = 0

    for batch in tqdm(loader, leave=False):
        left_eye = batch['left_eye'].to(device)
        right_eye = batch['right_eye'].to(device)
        pose = batch['pose'].to(device)
        target = batch['gaze'].to(device)

        with torch.set_grad_enabled(is_train):
            if scaler is not None and is_train:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(left_eye, right_eye, pose)
                    loss = loss_fn(pred, target)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(left_eye, right_eye, pose)
                loss = loss_fn(pred, target)
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

        ang = angular_error_deg(pred.detach(), target)
        n = target.shape[0]
        total_loss += loss.item() * n
        total_ang += ang.mean().item() * n
        total_n += n

    return total_loss / total_n, total_ang / total_n


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    set_seed(cfg['seed'])
    ensure_dir(cfg['save_dir'])

    train_subjects, val_subjects = split_subjects(cfg['data_root'], cfg['holdout_subject'])
    train_refs = MPIIGazeNormalizedDataset.build_refs(cfg['data_root'], train_subjects)
    val_refs = MPIIGazeNormalizedDataset.build_refs(cfg['data_root'], val_subjects)

    resize_to = (224, 224) if args.model_type == 'gazecapture_ft' else None
    train_ds = MPIIGazeNormalizedDataset(
        cfg['data_root'],
        train_refs,
        augment=True,
        resize_to=resize_to,
        aug_noise_std=cfg.get('aug_noise_std', 0.01),
        aug_brightness=cfg.get('aug_brightness', 0.0),
        aug_contrast=cfg.get('aug_contrast', 0.0),
    )
    val_ds = MPIIGazeNormalizedDataset(cfg['data_root'], val_refs, augment=False, resize_to=resize_to)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == 'gazecapture_ft':
        model = GazeCaptureEyeFTModel(hidden_dim=cfg['hidden_dim']).to(device)
        if args.gazecapture_ckpt:
            load_gazecapture_eye_weights(model, args.gazecapture_ckpt)
    else:
        model = GazeRegressor(hidden_dim=cfg['hidden_dim']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda') if (cfg.get('use_amp', True) and device.type == 'cuda') else None

    best_val = 1e9
    patience = int(cfg.get('early_stopping_patience', 1000))
    min_delta = float(cfg.get('early_stopping_min_delta', 0.0))
    epochs_no_improve = 0
    freeze_eye_epochs = int(cfg.get('freeze_eye_epochs', 0)) if args.model_type == 'gazecapture_ft' else 0
    for epoch in range(1, cfg['epochs'] + 1):
        if args.model_type == 'gazecapture_ft':
            set_eye_backbone_trainable(model, trainable=(epoch > freeze_eye_epochs))
        tr_loss, tr_ang = run_epoch(model, train_loader, optimizer, device, scaler)
        va_loss, va_ang = run_epoch(model, val_loader, None, device)
        eye_status = 'unfrozen' if (args.model_type != 'gazecapture_ft' or epoch > freeze_eye_epochs) else 'frozen'
        print(f'epoch={epoch:03d} eye={eye_status} train_loss={tr_loss:.5f} train_ang={tr_ang:.3f} val_loss={va_loss:.5f} val_ang={va_ang:.3f}')

        ckpt = {
            'model': model.state_dict(),
            'config': cfg,
            'epoch': epoch,
            'val_ang': va_ang,
        }
        torch.save(ckpt, Path(cfg['save_dir']) / 'last.pt')
        if va_ang < (best_val - min_delta):
            best_val = va_ang
            epochs_no_improve = 0
            torch.save(ckpt, Path(cfg['save_dir']) / 'best.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'early_stopping at epoch={epoch:03d} best_val_angular_error_deg={best_val:.3f}')
                break

    print(f'best_val_angular_error_deg={best_val:.3f}')


if __name__ == '__main__':
    main()
