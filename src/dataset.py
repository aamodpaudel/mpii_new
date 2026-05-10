from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from .utils import gaze_vec_to_angles


@dataclass(frozen=True)
class SampleRef:
    subject: str
    day_file: str
    idx: int


class MPIIGazeNormalizedDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        refs: List[SampleRef],
        augment: bool = False,
        resize_to: Tuple[int, int] | None = None,
        aug_noise_std: float = 0.01,
        aug_brightness: float = 0.0,
        aug_contrast: float = 0.0,
    ):
        self.data_root = Path(data_root)
        self.refs = refs
        self.augment = augment
        self.resize_to = resize_to
        self.aug_noise_std = aug_noise_std
        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast
        self._cache: Dict[Path, object] = {}

    @staticmethod
    def build_refs(data_root: str | Path, subjects: List[str]) -> List[SampleRef]:
        root = Path(data_root)
        refs: List[SampleRef] = []
        for subject in subjects:
            subject_dir = root / subject
            if not subject_dir.exists():
                continue
            for day_file in sorted(subject_dir.glob('day*.mat')):
                mat = sio.loadmat(day_file, squeeze_me=True, struct_as_record=False)
                data = mat['data']
                n_left = MPIIGazeNormalizedDataset._count_samples(data.left.image, data.left.gaze, data.left.pose)
                n_right = MPIIGazeNormalizedDataset._count_samples(data.right.image, data.right.gaze, data.right.pose)
                n = min(n_left, n_right)
                if n <= 0:
                    continue
                refs.extend(
                    SampleRef(subject=subject, day_file=day_file.name, idx=i)
                    for i in range(n)
                )
        return refs

    def __len__(self) -> int:
        return len(self.refs)

    def _load_day(self, subject: str, day_file: str):
        path = self.data_root / subject / day_file
        if path not in self._cache:
            self._cache[path] = sio.loadmat(path, squeeze_me=True, struct_as_record=False)['data']
        return self._cache[path]

    def __getitem__(self, i: int):
        ref = self.refs[i]
        data = self._load_day(ref.subject, ref.day_file)
        left_eye = self._get_sample_image(data.left.image, ref.idx).astype(np.float32) / 255.0
        right_eye = self._get_sample_image(data.right.image, ref.idx).astype(np.float32) / 255.0
        gaze_vec = self._get_sample_vec3(data.left.gaze, ref.idx).astype(np.float32)
        pose_vec = self._get_sample_vec3(data.left.pose, ref.idx).astype(np.float32)

        if self.augment:
            if self.aug_contrast > 0:
                c = 1.0 + np.random.uniform(-self.aug_contrast, self.aug_contrast)
                left_eye = np.clip((left_eye - 0.5) * c + 0.5, 0.0, 1.0)
                right_eye = np.clip((right_eye - 0.5) * c + 0.5, 0.0, 1.0)
            if self.aug_brightness > 0:
                b = np.random.uniform(-self.aug_brightness, self.aug_brightness)
                left_eye = np.clip(left_eye + b, 0.0, 1.0)
                right_eye = np.clip(right_eye + b, 0.0, 1.0)
            left_eye = left_eye + np.random.normal(0.0, self.aug_noise_std, left_eye.shape).astype(np.float32)
            right_eye = right_eye + np.random.normal(0.0, self.aug_noise_std, right_eye.shape).astype(np.float32)
            left_eye = np.clip(left_eye, 0.0, 1.0)
            right_eye = np.clip(right_eye, 0.0, 1.0)
        if self.resize_to is not None:
            w, h = self.resize_to
            left_eye = cv2.resize(left_eye, (w, h), interpolation=cv2.INTER_LINEAR)
            right_eye = cv2.resize(right_eye, (w, h), interpolation=cv2.INTER_LINEAR)

        gaze_angles = gaze_vec_to_angles(gaze_vec)
        left_eye = np.expand_dims(left_eye, axis=0)
        right_eye = np.expand_dims(right_eye, axis=0)

        return {
            'left_eye': torch.from_numpy(left_eye),
            'right_eye': torch.from_numpy(right_eye),
            'pose': torch.from_numpy(pose_vec),
            'gaze': torch.from_numpy(gaze_angles.astype(np.float32)),
            'subject': ref.subject,
        }

    @staticmethod
    def _get_sample_image(arr: np.ndarray, idx: int) -> np.ndarray:
        # Handles [N,H,W] and singleton [H,W].
        if arr.ndim == 3:
            return arr[idx]
        if arr.ndim == 2:
            if idx != 0:
                raise IndexError(f'image idx {idx} out of bounds for singleton image')
            return arr
        raise ValueError(f'unexpected image shape: {arr.shape}')

    @staticmethod
    def _get_sample_vec3(arr: np.ndarray, idx: int) -> np.ndarray:
        # Handles [N,3], [3,N], and singleton [3].
        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError(f'unexpected vec shape: {arr.shape}')
            if idx != 0:
                raise IndexError(f'vec idx {idx} out of bounds for singleton vec')
            return arr
        if arr.ndim == 2:
            if arr.shape[1] == 3:  # [N,3]
                return arr[idx]
            if arr.shape[0] == 3:  # [3,N]
                return arr[:, idx]
        raise ValueError(f'unexpected vec shape: {arr.shape}')

    @staticmethod
    def _infer_n_image(arr: np.ndarray) -> int:
        if arr.ndim == 3:  # [N,H,W]
            return int(arr.shape[0])
        if arr.ndim == 2:  # singleton [H,W]
            return 1
        return 0

    @staticmethod
    def _infer_n_vec3(arr: np.ndarray) -> int:
        if arr.ndim == 1 and arr.shape[0] == 3:
            return 1
        if arr.ndim == 2:
            if arr.shape[1] == 3:  # [N,3]
                return int(arr.shape[0])
            if arr.shape[0] == 3:  # [3,N]
                return int(arr.shape[1])
        return 0

    @staticmethod
    def _count_samples(image_arr: np.ndarray, gaze_arr: np.ndarray, pose_arr: np.ndarray) -> int:
        return min(
            MPIIGazeNormalizedDataset._infer_n_image(image_arr),
            MPIIGazeNormalizedDataset._infer_n_vec3(gaze_arr),
            MPIIGazeNormalizedDataset._infer_n_vec3(pose_arr),
        )


def split_subjects(data_root: str | Path, holdout_subject: str) -> Tuple[List[str], List[str]]:
    subjects = sorted([p.name for p in Path(data_root).iterdir() if p.is_dir() and p.name.startswith('p')])
    train_subjects = [s for s in subjects if s != holdout_subject]
    val_subjects = [holdout_subject]
    return train_subjects, val_subjects
