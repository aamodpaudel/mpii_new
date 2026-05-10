from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pygame
import torch
from PIL import Image, ImageSequence

from src.model import GazeRegressor
from src.model_gazecapture_ft import GazeCaptureEyeFTModel

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None


@dataclass
class FaceState:
    left_eye: np.ndarray
    right_eye: np.ndarray
    pose: np.ndarray


def load_model(checkpoint: Path, model_type: str):
    ckpt = torch.load(checkpoint, map_location='cpu')
    cfg = ckpt.get('config', {})
    hidden_dim = int(cfg.get('hidden_dim', 256))
    if model_type == 'gazecapture_ft':
        model = GazeCaptureEyeFTModel(hidden_dim=hidden_dim)
        eye_size = (224, 224)
    else:
        model = GazeRegressor(hidden_dim=hidden_dim)
        eye_size = (60, 36)  # w,h
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, eye_size


def crop_eye(gray: np.ndarray, pts: np.ndarray, out_wh: Tuple[int, int]) -> np.ndarray:
    h, w = gray.shape[:2]
    x0 = int(np.clip(np.min(pts[:, 0]) - 6, 0, w - 1))
    x1 = int(np.clip(np.max(pts[:, 0]) + 6, 0, w - 1))
    y0 = int(np.clip(np.min(pts[:, 1]) - 6, 0, h - 1))
    y1 = int(np.clip(np.max(pts[:, 1]) + 6, 0, h - 1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((out_wh[1], out_wh[0]), dtype=np.float32)
    eye = gray[y0:y1, x0:x1]
    eye = cv2.resize(eye, out_wh, interpolation=cv2.INTER_LINEAR)
    return eye.astype(np.float32) / 255.0


def extract_face_state_mediapipe(frame_bgr: np.ndarray, face_mesh, out_wh: Tuple[int, int]) -> FaceState | None:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    h, w = frame_bgr.shape[:2]

    def pt(i: int):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

    left_idx = [33, 133, 159, 145, 153, 154]
    right_idx = [362, 263, 386, 374, 380, 381]
    left_pts = np.stack([pt(i) for i in left_idx], axis=0)
    right_pts = np.stack([pt(i) for i in right_idx], axis=0)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    left_eye = crop_eye(gray, left_pts, out_wh)
    right_eye = crop_eye(gray, right_pts, out_wh)

    left_center = left_pts.mean(axis=0)
    right_center = right_pts.mean(axis=0)
    nose = pt(1)
    eye_mid = (left_center + right_center) / 2.0
    eye_dist = np.linalg.norm(right_center - left_center) + 1e-6

    yaw = float((nose[0] - eye_mid[0]) / eye_dist)
    pitch = float((nose[1] - eye_mid[1]) / eye_dist)
    roll = float(math.atan2((right_center[1] - left_center[1]), (right_center[0] - left_center[0])))
    pose = np.array([yaw, pitch, roll], dtype=np.float32)

    return FaceState(left_eye=left_eye, right_eye=right_eye, pose=pose)


def extract_face_state_haar(
    frame_bgr: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
    out_wh: Tuple[int, int],
) -> FaceState | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(120, 120))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda x: x[2] * x[3])
    face_roi = gray[fy : fy + fh, fx : fx + fw]
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.15, minNeighbors=6, minSize=(20, 20))
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes[0], eyes[1]

    lpts = np.array(
        [[fx + ex1, fy + ey1], [fx + ex1 + ew1, fy + ey1], [fx + ex1, fy + ey1 + eh1], [fx + ex1 + ew1, fy + ey1 + eh1]],
        dtype=np.float32,
    )
    rpts = np.array(
        [[fx + ex2, fy + ey2], [fx + ex2 + ew2, fy + ey2], [fx + ex2, fy + ey2 + eh2], [fx + ex2 + ew2, fy + ey2 + eh2]],
        dtype=np.float32,
    )

    left_eye = crop_eye(gray, lpts, out_wh)
    right_eye = crop_eye(gray, rpts, out_wh)

    lc = lpts.mean(axis=0)
    rc = rpts.mean(axis=0)
    eye_mid = (lc + rc) / 2.0
    eye_dist = np.linalg.norm(rc - lc) + 1e-6
    face_center = np.array([fx + fw * 0.5, fy + fh * 0.5], dtype=np.float32)
    yaw = float((eye_mid[0] - face_center[0]) / eye_dist)
    pitch = float((eye_mid[1] - face_center[1]) / eye_dist)
    roll = float(math.atan2((rc[1] - lc[1]), (rc[0] - lc[0])))
    pose = np.array([yaw, pitch, roll], dtype=np.float32)
    return FaceState(left_eye=left_eye, right_eye=right_eye, pose=pose)


def predict_angles(model, state: FaceState, device: torch.device) -> np.ndarray:
    l = torch.from_numpy(state.left_eye).unsqueeze(0).unsqueeze(0).to(device)
    r = torch.from_numpy(state.right_eye).unsqueeze(0).unsqueeze(0).to(device)
    p = torch.from_numpy(state.pose).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(l, r, p).squeeze(0).cpu().numpy()
    return out.astype(np.float32)


def fit_affine(src: np.ndarray, dst: np.ndarray):
    x = np.concatenate([src, np.ones((src.shape[0], 1), dtype=np.float32)], axis=1)
    w, *_ = np.linalg.lstsq(x, dst, rcond=None)
    return w.astype(np.float32)


def apply_affine(w: np.ndarray, gaze_angles: np.ndarray):
    x = np.array([gaze_angles[0], gaze_angles[1], 1.0], dtype=np.float32)
    y = x @ w
    return np.clip(y, 0.0, 1.0)


def load_gif_frames(path: Path, target_h: int = 110) -> List[pygame.Surface]:
    frames: List[pygame.Surface] = []
    with Image.open(path) as im:
        for f in ImageSequence.Iterator(im):
            rgba = f.convert('RGBA')
            w, h = rgba.size
            scale = target_h / float(h)
            nw, nh = max(1, int(w * scale)), target_h
            rgba = rgba.resize((nw, nh), Image.Resampling.BILINEAR)
            mode = rgba.mode
            data = rgba.tobytes()
            surf = pygame.image.fromstring(data, rgba.size, mode).convert_alpha()
            frames.append(surf)
    return frames


def open_webcam(camera_index: int, backend_name: str = 'auto'):
    backends = []
    name = backend_name.lower()
    if name == 'auto':
        backends = [('MSMF', cv2.CAP_MSMF), ('DSHOW', cv2.CAP_DSHOW), ('DEFAULT', None)]
    elif name == 'msmf':
        backends = [('MSMF', cv2.CAP_MSMF)]
    elif name == 'dshow':
        backends = [('DSHOW', cv2.CAP_DSHOW)]
    else:
        backends = [('DEFAULT', None)]

    tried = []
    candidates = [camera_index]
    if camera_index != 0:
        candidates.append(0)
    if camera_index != 1:
        candidates.append(1)

    for idx in candidates:
        for label, backend in backends:
            cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
            ok = bool(cap is not None and cap.isOpened())
            tried.append(f'index={idx}, backend={label}, opened={ok}')
            if ok:
                # Validate we can read at least one frame.
                ret, _ = cap.read()
                if ret:
                    print(f'[camera] Opened webcam at index={idx} backend={label}')
                    return cap
                cap.release()
                tried.append(f'index={idx}, backend={label}, read_first_frame=False')
            else:
                if cap is not None:
                    cap.release()

    details = '\n'.join(tried)
    raise RuntimeError(
        'Could not open webcam.\n'
        'Tried combinations:\n'
        f'{details}\n'
        'Tip: close apps using camera (Zoom/Teams/Camera), then retry with --camera 0 or 1 '
        'and optionally --camera_backend dshow or --camera_backend msmf.'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default='runs/mpii_ft_tuned/best.pt')
    ap.add_argument('--model_type', type=str, default='gazecapture_ft', choices=['gazecapture_ft', 'baseline'])
    ap.add_argument('--gif', type=str, default='d:/mpii/bird_flying.gif')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--camera_backend', type=str, default='auto', choices=['auto', 'dshow', 'msmf', 'default'])
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--calib_mode', type=str, default='four_dirs', choices=['four_dirs', 'grid'])
    ap.add_argument('--calib_grid', type=int, default=4, help='grid size K -> KxK points (used when --calib_mode grid)')
    ap.add_argument('--calib_samples_per_point', type=int, default=20)
    ap.add_argument('--cursor_smooth_alpha', type=float, default=0.18)
    ap.add_argument('--cursor_deadzone_px', type=float, default=8.0)
    ap.add_argument('--bird_speed_scale', type=float, default=0.55)
    args = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption('EyeOS Bird Shooter (Research Demo)')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('consolas', 24)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, eye_wh = load_model(Path(args.checkpoint), args.model_type)
    model.to(device)

    cap = open_webcam(args.camera, args.camera_backend)

    use_mediapipe = bool(mp is not None and hasattr(mp, 'solutions'))
    mp_face = None
    if use_mediapipe:
        mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    frames = load_gif_frames(Path(args.gif))
    frame_idx = 0
    bird_x, bird_y = 40.0, args.height * 0.3
    bird_vx, bird_vy = 280.0 * args.bird_speed_scale, 140.0 * args.bird_speed_scale

    # Focused calibration for gameplay region.
    if args.calib_mode == 'four_dirs':
        # 8-direction calibration points: left, right, top, bottom + diagonals.
        calib_points: List[Tuple[float, float]] = [
            (0.12, 0.50),
            (0.88, 0.50),
            (0.50, 0.20),
            (0.50, 0.80),
            (0.20, 0.20),  # NW
            (0.80, 0.20),  # NE
            (0.20, 0.80),  # SW
            (0.80, 0.80),  # SE
        ]
    else:
        margin = 0.08
        xs = np.linspace(margin, 1.0 - margin, args.calib_grid, dtype=np.float32)
        ys = np.linspace(margin, 1.0 - margin, args.calib_grid, dtype=np.float32)
        calib_points = []
        for ridx, y in enumerate(ys):
            row = [(float(x), float(y)) for x in xs]
            if ridx % 2 == 1:
                row.reverse()
            calib_points.extend(row)

    calib_i = 0
    calib_gaze: List[np.ndarray] = []
    calib_screen: List[np.ndarray] = []
    recent_preds: List[np.ndarray] = []
    point_preds: List[np.ndarray] = []
    mapper = None

    score = 0
    last_shot = 0.0
    gaze_cursor = np.array([0.5, 0.5], dtype=np.float32)
    raw_cursor = np.array([0.5, 0.5], dtype=np.float32)
    cursor_hist = deque(maxlen=7)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                if mapper is None and ev.key == pygame.K_SPACE and point_preds:
                    avg_pred = np.mean(np.stack(point_preds, axis=0), axis=0)
                    calib_gaze.append(avg_pred)
                    calib_screen.append(np.array(calib_points[calib_i], dtype=np.float32))
                    calib_i += 1
                    point_preds = []
                    recent_preds = []
                    if calib_i >= len(calib_points):
                        mapper = fit_affine(np.stack(calib_gaze), np.stack(calib_screen))
                elif mapper is not None and ev.key == pygame.K_SPACE:
                    now = time.time()
                    if now - last_shot > 0.15:
                        last_shot = now
                        bx = int(bird_x + frames[frame_idx].get_width() // 2)
                        by = int(bird_y + frames[frame_idx].get_height() // 2)
                        cx = int(gaze_cursor[0] * args.width)
                        cy = int(gaze_cursor[1] * args.height)
                        if math.hypot(cx - bx, cy - by) < 80:
                            score += 1
                            bird_x = -frames[0].get_width()
                            bird_y = np.random.uniform(40, args.height - 180)

        ok, frame = cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            if use_mediapipe and mp_face is not None:
                state = extract_face_state_mediapipe(frame, mp_face, eye_wh)
            else:
                state = extract_face_state_haar(frame, face_cascade, eye_cascade, eye_wh)
            if state is not None:
                pred = predict_angles(model, state, device)
                recent_preds.append(pred)
                if len(recent_preds) > 12:
                    recent_preds.pop(0)
                if mapper is None:
                    point_preds.append(pred)
                    if len(point_preds) > args.calib_samples_per_point:
                        point_preds.pop(0)
                else:
                    mapped = apply_affine(mapper, pred)
                    raw_cursor = mapped
                    cursor_hist.append(raw_cursor.copy())
                    med = np.median(np.stack(cursor_hist, axis=0), axis=0)
                    smoothed = gaze_cursor * (1.0 - args.cursor_smooth_alpha) + med * args.cursor_smooth_alpha
                    dx = (smoothed[0] - gaze_cursor[0]) * args.width
                    dy = (smoothed[1] - gaze_cursor[1]) * args.height
                    if (dx * dx + dy * dy) >= (args.cursor_deadzone_px * args.cursor_deadzone_px):
                        gaze_cursor = smoothed

        bird_x += bird_vx * dt
        bird_y += bird_vy * dt
        if bird_x > args.width + 50:
            bird_x = -frames[0].get_width()
            bird_y = np.random.uniform(40, args.height - 180)
        if bird_y < 20 or bird_y > args.height - 140:
            bird_vy *= -1

        screen.fill((255, 255, 255))

        if mapper is None:
            px, py = calib_points[calib_i]
            tx, ty = int(px * args.width), int(py * args.height)
            pygame.draw.circle(screen, (40, 40, 40), (tx, ty), 16)
            pygame.draw.circle(screen, (0, 0, 0), (tx, ty), 4)
            backend = 'MediaPipe' if use_mediapipe else 'OpenCV Haar'
            ready_n = min(len(point_preds), args.calib_samples_per_point)
            msg = (
                f'Calibration {calib_i+1}/{len(calib_points)} ({backend}) '
                f'collecting {ready_n}/{args.calib_samples_per_point} samples, then SPACE'
            )
            screen.blit(font.render(msg, True, (0, 0, 0)), (30, 25))
        else:
            frame_idx = (frame_idx + 1) % len(frames)
            bird = frames[frame_idx]
            screen.blit(bird, (int(bird_x), int(bird_y)))

            cx = int(gaze_cursor[0] * args.width)
            cy = int(gaze_cursor[1] * args.height)
            pygame.draw.circle(screen, (220, 20, 20), (cx, cy), 10, width=2)
            pygame.draw.line(screen, (220, 20, 20), (cx - 16, cy), (cx + 16, cy), width=2)
            pygame.draw.line(screen, (220, 20, 20), (cx, cy - 16), (cx, cy + 16), width=2)

            screen.blit(font.render('SPACE = shoot, ESC = exit', True, (0, 0, 0)), (30, 25))
            screen.blit(font.render(f'Score: {score}', True, (0, 0, 0)), (30, 58))

        pygame.display.flip()

    cap.release()
    if mp_face is not None:
        mp_face.close()
    pygame.quit()


if __name__ == '__main__':
    main()
