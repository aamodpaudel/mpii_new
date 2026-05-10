from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaze_vec_to_angles(g: np.ndarray) -> np.ndarray:
    # Matches MPIIGaze normalization convention.
    x, y, z = g[..., 0], g[..., 1], g[..., 2]
    theta = np.arcsin(np.clip(-y, -1.0, 1.0))
    phi = np.arctan2(-x, -z)
    return np.stack([theta, phi], axis=-1)


def angles_to_unit_vector(theta_phi: torch.Tensor) -> torch.Tensor:
    theta = theta_phi[..., 0]
    phi = theta_phi[..., 1]
    x = -torch.cos(theta) * torch.sin(phi)
    y = -torch.sin(theta)
    z = -torch.cos(theta) * torch.cos(phi)
    v = torch.stack([x, y, z], dim=-1)
    return torch.nn.functional.normalize(v, dim=-1)


def angular_error_deg(pred_angles: torch.Tensor, true_angles: torch.Tensor) -> torch.Tensor:
    pv = angles_to_unit_vector(pred_angles)
    tv = angles_to_unit_vector(true_angles)
    cos_sim = (pv * tv).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_sim))


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
