# EyeOS Training Starter (MPIIGaze-first)

This project is a practical starter for training an eye-gaze model from `MPIIGaze` and then adding personalization.

## Important license note

`MPIIGaze` is CC BY-NC-SA 4.0 (non-commercial). Do not use this dataset for commercial model training/deployment.

## What this starter includes

- `train.py`: train a two-eye + head-pose gaze regressor (outputs `(theta, phi)`).
- `src/dataset.py`: loads `MPIIGaze/Data/Normalized/*/day*.mat`.
- `src/model.py`: dual eye encoders + fusion MLP.
- `calibrate.py`: user-specific calibration adapter (angles -> screen `(x,y)`) with online update support.
- `infer.py`: minimal inference test utility.

## Recommended architecture for EyeOS v1

1. Base model: train on large public data for generalization.
2. Personal adapter: 30-second calibration maps model angles to per-user screen coordinates.
3. Continuous learning: update only adapter online from confirmed click points.

This mirrors production systems: stable foundation + lightweight personalization.

## Datasets and order

1. `GazeCapture` (first): largest in-the-wild generalization source.
2. `MPIIGaze` (second): laptop webcam domain adaptation.
3. `ETH-XGaze` (third): robustness to wider head poses.
4. `UnityEyes` (optional): synthetic augmentation for tail cases.

If you only have MPII today, start there (this repo), then fine-tune from a GazeCapture-pretrained checkpoint when available.

## Quick start

From `d:\mpii\eyeos_train`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Edit `configs/mpii_baseline.yaml` if needed:
- `data_root`: currently `d:/mpii/MPIIGaze/Data/Normalized`
- `holdout_subject`: subject used for validation (LOSO style)

Train:

```powershell
python train.py --config configs/mpii_baseline.yaml
```

Fine-tune from GazeCapture pretrained eye pathway (recommended next step):

```powershell
python train.py --config configs/mpii_baseline.yaml --model_type gazecapture_ft --gazecapture_ckpt d:/mpii/GazeCapture/pytorch/checkpoint.pth.tar
```

Outputs:
- `runs/mpii_baseline/last.pt`
- `runs/mpii_baseline/best.pt`

## Metrics to track

- Main metric: mean angular error (degrees).
- Early targets:
  - `MPIIGaze-only`: around 5-7 deg is common for a first baseline.
  - with bigger pretraining + personalization: can drop further.

## 30-second calibration flow

Collect a small `.npz` with:
- `gaze_angles`: shape `[N,2]` from model predictions during calibration dots.
- `screen_xy`: shape `[N,2]` normalized screen coords in `[0,1]`.

Fit adapter:

```powershell
python calibrate.py --calib_npz calibration_user01.npz --out user01_adapter.joblib
```

Online update during usage:
- whenever user confirms intent (e.g., blink-click), append `(gaze_angles_t, cursor_xy_t)`
- call `partial_fit` on adapter in small batches every few seconds.

## Production roadmap

1. Replace static eye crops with real-time MediaPipe face mesh extraction.
2. Add full-face stream and face-grid stream to improve robustness.
3. Export model to ONNX:

```powershell
python -c "import torch; from src.model import GazeRegressor; m=GazeRegressor(); m.eval(); l=torch.randn(1,1,36,60); r=torch.randn(1,1,36,60); p=torch.randn(1,3); torch.onnx.export(m,(l,r,p),'gaze_model.onnx',input_names=['left_eye','right_eye','pose'],output_names=['theta_phi'],opset_version=17); print('saved gaze_model.onnx')"
```

4. Run ONNX in browser/backend; keep personalization adapter user-local.

## Notes on the 97% claim

For accessibility use, report transparent metrics:
- angular error (deg)
- pointing error in pixels/mm
- task completion rate in target selection tasks

"97% accuracy" needs a strict definition (for example, percent of dwell targets hit within radius/time).

## File map

- `train.py`
- `calibrate.py`
- `infer.py`
- `src/dataset.py`
- `src/model.py`
- `src/utils.py`
- `configs/mpii_baseline.yaml`
- `requirements.txt`
