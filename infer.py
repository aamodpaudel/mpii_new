from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.model import GazeRegressor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--left_eye_npy', type=str, required=True)
    p.add_argument('--right_eye_npy', type=str, required=True)
    p.add_argument('--pose_npy', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = ckpt['config']

    model = GazeRegressor(hidden_dim=cfg['hidden_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()

    l = np.load(args.left_eye_npy).astype(np.float32)
    r = np.load(args.right_eye_npy).astype(np.float32)
    p = np.load(args.pose_npy).astype(np.float32)

    l = torch.from_numpy(l).unsqueeze(0).unsqueeze(0)
    r = torch.from_numpy(r).unsqueeze(0).unsqueeze(0)
    p = torch.from_numpy(p).unsqueeze(0)

    with torch.no_grad():
        gaze = model(l, r, p).squeeze(0).numpy()

    print('pred_theta_phi_rad=', gaze.tolist())


if __name__ == '__main__':
    main()
