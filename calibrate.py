from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class OnlineScreenAdapter:
    # Maps model gaze angles -> screen coordinates and supports partial_fit updates.
    def __init__(self):
        base = SGDRegressor(loss='squared_error', penalty='l2', alpha=1e-4, learning_rate='invscaling', eta0=0.01)
        self.model = MultiOutputRegressor(make_pipeline(StandardScaler(), base))
        self.is_fit = False

    def fit(self, gaze_angles: np.ndarray, screen_xy: np.ndarray):
        self.model.fit(gaze_angles, screen_xy)
        self.is_fit = True

    def partial_fit(self, gaze_angles: np.ndarray, screen_xy: np.ndarray):
        if not self.is_fit:
            self.fit(gaze_angles, screen_xy)
        else:
            self.model.partial_fit(gaze_angles, screen_xy)

    def predict(self, gaze_angles: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError('Adapter is not fit yet.')
        return self.model.predict(gaze_angles)

    def save(self, path: str | Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> 'OnlineScreenAdapter':
        return joblib.load(path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--calib_npz', type=str, required=True, help='npz with gaze_angles [N,2], screen_xy [N,2]')
    p.add_argument('--out', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    d = np.load(args.calib_npz)
    adapter = OnlineScreenAdapter()
    adapter.fit(d['gaze_angles'], d['screen_xy'])
    adapter.save(args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
