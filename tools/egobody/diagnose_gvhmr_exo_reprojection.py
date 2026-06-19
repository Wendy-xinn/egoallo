#!/usr/bin/env python3
"""Diagnose GVHMR exo export in the EgoBody/EgoAllo world.

This script checks the coordinate chain used by export_gvhmr_exo_for_egoallo.py:

    GVHMR smpl_params_incam -> exo vertices_world -> inverse T_world_pv -> image reprojection

It writes:
  - metrics.json with reprojection visibility and root trajectory statistics
  - root_trajectory.png comparing exported exo root, interactee GT transl, and PV camera
  - reprojection_*.jpg overlays of exported exo mesh/joints projected back to EgoBody images

The GT root comparison is intentionally lightweight: it uses EgoBody interactee SMPL-X
`transl` from pkl files, transformed into the same z-up world. It does not require
loading the SMPL-X body model.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

BODY22 = np.arange(22, dtype=np.int64)
HEAD_JOINT = 15
LEFT_WRIST_JOINT = 20
RIGHT_WRIST_JOINT = 21


def _np_scalar_to_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def _camera_to_pv_matrix(mode: str) -> np.ndarray:
    if mode == "opencv_to_holo":
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    if mode == "identity":
        return np.eye(3, dtype=np.float32)
    raise ValueError(f"Unknown camera convention: {mode}")


def _axis_conversion_matrix() -> np.ndarray:
    # HoloLens y-up -> EgoAllo z-up: x'=x, y'=-z, z'=y.
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )


def _make_transform(rotation: np.ndarray, translation: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rotation.astype(np.float32)
    if translation is not None:
        T[:3, 3] = np.asarray(translation, dtype=np.float32)
    return T


def _load_kinect12_to_holo(egobody_root: Path, recording: str) -> np.ndarray:
    calib_path = egobody_root / "calibrations" / recording / "cal_trans" / "holo_to_kinect12.json"
    with calib_path.open("r") as f:
        holo_to_kinect = np.asarray(json.load(f)["trans"], dtype=np.float32)
    return np.linalg.inv(holo_to_kinect).astype(np.float32)


def _gt_points_transform(egobody_root: Path, recording: str, mode: str) -> np.ndarray:
    if mode == "none":
        return np.eye(4, dtype=np.float32)
    if mode == "kinect12_to_holo":
        return _load_kinect12_to_holo(egobody_root, recording)
    if mode == "holo_y_up_to_z_up":
        return _make_transform(_axis_conversion_matrix())
    if mode == "kinect12_to_holo_zup":
        return _make_transform(_axis_conversion_matrix()) @ _load_kinect12_to_holo(egobody_root, recording)
    raise ValueError(f"Unknown GT coordinate mode: {mode}")


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    flat = points.reshape(-1, 3)
    out = flat @ T[:3, :3].T + T[:3, 3]
    return out.reshape(points.shape).astype(np.float32)


def _parse_frame_id(path_like: object) -> int | None:
    stem = Path(str(path_like)).stem
    match = re.search(r"_frame_(\d+)$", stem)
    if match is None:
        match = re.search(r"frame_(\d+)$", stem)
    if match is None:
        return None
    return int(match.group(1))


def _find_gt_recording_root(egobody_root: Path, recording: str, split: str, role: str) -> Path | None:
    role_prefix = {
        "camera_wearer": "smplx_camera_wearer",
        "interactee": "smplx_interactee",
    }[role]
    splits = [split] if split != "auto" else ["train", "val", "test"]
    for split_name in splits:
        candidate = egobody_root / f"{role_prefix}_{split_name}" / recording
        if candidate.exists():
            return candidate
    return None


def _find_gt_frame_map(gt_root: Path, body_idx: int | None) -> dict[int, Path]:
    body_dirs = sorted(gt_root.glob("body_idx_*")) if body_idx is None else [gt_root / f"body_idx_{body_idx}"]
    for body_dir in body_dirs:
        if not body_dir.exists():
            continue
        frame_map: dict[int, Path] = {}
        for p in body_dir.glob("results/frame_*/000.pkl"):
            frame_id = _parse_frame_id(p.parent.name)
            if frame_id is not None:
                frame_map[frame_id] = p
        if frame_map:
            return frame_map
    return {}


def _load_gt_transl_sequence(
    egobody_root: Path,
    recording: str,
    split: str,
    role: str,
    body_idx: int | None,
    frame_ids: np.ndarray,
    gt_coordinate_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    gt_root = _find_gt_recording_root(egobody_root, recording, split, role)
    if gt_root is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)
    frame_map = _find_gt_frame_map(gt_root, body_idx)
    if not frame_map:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)

    T_gt = _gt_points_transform(egobody_root, recording, gt_coordinate_mode)
    roots = []
    valid = []
    for frame_id in frame_ids:
        path = frame_map.get(int(frame_id))
        if path is None:
            roots.append(np.zeros(3, dtype=np.float32))
            valid.append(False)
            continue
        with path.open("rb") as f:
            data = pickle.load(f)
        transl = np.asarray(data.get("transl", np.zeros((1, 3))), dtype=np.float32).reshape(-1, 3)[0]
        roots.append(transl)
        valid.append(np.isfinite(transl).all())
    roots_np = np.asarray(roots, dtype=np.float32)
    valid_np = np.asarray(valid, dtype=bool)
    if len(roots_np) > 0:
        roots_np = _transform_points(roots_np, T_gt)
    return roots_np, valid_np


def _stack_gt_param(frames: list[dict[str, Any]], key: str, width: int, device, default: float = 0.0):
    import torch

    values = []
    for frame in frames:
        if key in frame:
            arr = np.asarray(frame[key], dtype=np.float32).reshape(-1, width)[0]
        else:
            arr = np.full((width,), default, dtype=np.float32)
        values.append(arr)
    return torch.from_numpy(np.stack(values, axis=0)).to(device=device)


def _load_gt_joints_sequence(
    egobody_root: Path,
    recording: str,
    split: str,
    role: str,
    body_idx: int | None,
    frame_ids: np.ndarray,
    gt_coordinate_mode: str,
    smplx_model_path: Path,
    device_name: str,
    zero_betas: bool,
) -> tuple[np.ndarray, np.ndarray]:
    gt_root = _find_gt_recording_root(egobody_root, recording, split, role)
    if gt_root is None:
        return np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)
    frame_map = _find_gt_frame_map(gt_root, body_idx)
    if not frame_map:
        return np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)

    pkl_paths: list[Path | None] = [frame_map.get(int(frame_id)) for frame_id in frame_ids]
    valid = np.asarray([p is not None for p in pkl_paths], dtype=bool)
    if not np.any(valid):
        return np.zeros((len(frame_ids), 0, 3), dtype=np.float32), valid

    try:
        import torch
        import smplx
    except Exception as exc:
        print(f"[warn] Cannot import torch/smplx for GT joint metrics: {exc}")
        return np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    frames = []
    valid_indices = []
    for i, path in enumerate(pkl_paths):
        if path is None:
            continue
        with path.open("rb") as f:
            frames.append(pickle.load(f))
        valid_indices.append(i)

    model = smplx.create(
        str(smplx_model_path),
        model_type="smplx",
        gender="neutral",
        use_pca=True,
        num_pca_comps=12,
        batch_size=len(frames),
    ).to(device)
    betas = _stack_gt_param(frames, "betas", 10, device)
    if zero_betas:
        betas = torch.zeros_like(betas)
    with torch.no_grad():
        out = model(
            betas=betas,
            global_orient=_stack_gt_param(frames, "global_orient", 3, device),
            transl=_stack_gt_param(frames, "transl", 3, device),
            body_pose=_stack_gt_param(frames, "body_pose", 63, device),
            left_hand_pose=_stack_gt_param(frames, "left_hand_pose", 12, device),
            right_hand_pose=_stack_gt_param(frames, "right_hand_pose", 12, device),
            jaw_pose=_stack_gt_param(frames, "jaw_pose", 3, device),
            leye_pose=_stack_gt_param(frames, "leye_pose", 3, device),
            reye_pose=_stack_gt_param(frames, "reye_pose", 3, device),
            expression=_stack_gt_param(frames, "expression", 10, device),
            return_verts=False,
        )
    joints_valid = out.joints.detach().cpu().numpy().astype(np.float32)
    T_gt = _gt_points_transform(egobody_root, recording, gt_coordinate_mode)
    joints_valid = _transform_points(joints_valid, T_gt)

    joints = np.zeros((len(frame_ids), joints_valid.shape[1], 3), dtype=np.float32)
    for src_i, dst_i in enumerate(valid_indices):
        joints[dst_i] = joints_valid[src_i]
    return joints, valid


def _batch_procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    aligned = np.zeros_like(pred)
    for i in range(len(pred)):
        X = pred[i]
        Y = gt[i]
        mu_x = X.mean(axis=0, keepdims=True)
        mu_y = Y.mean(axis=0, keepdims=True)
        X0 = X - mu_x
        Y0 = Y - mu_y
        norm_x = np.linalg.norm(X0)
        norm_y = np.linalg.norm(Y0)
        if norm_x < 1e-8 or norm_y < 1e-8:
            aligned[i] = X
            continue
        Xn = X0 / norm_x
        Yn = Y0 / norm_y
        U, _, Vt = np.linalg.svd(Xn.T @ Yn)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = U @ Vt
        scale = norm_y / norm_x
        aligned[i] = scale * X0 @ R + mu_y
    return aligned.astype(np.float32)


def _joint_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, joints: np.ndarray) -> dict[str, dict[str, float] | int]:
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[0] != gt.shape[0] or not np.any(valid):
        return {"valid_frames": 0}
    joints = np.asarray([j for j in joints if j < pred.shape[1] and j < gt.shape[1]], dtype=np.int64)
    if joints.size == 0:
        return {"valid_frames": 0}
    pred_v = pred[valid][:, joints]
    gt_v = gt[valid][:, joints]
    err = np.linalg.norm(pred_v - gt_v, axis=-1).mean(axis=-1)
    pred_ra = pred_v - pred_v[:, :1]
    gt_ra = gt_v - gt_v[:, :1]
    err_ra = np.linalg.norm(pred_ra - gt_ra, axis=-1).mean(axis=-1)
    pred_pa = _batch_procrustes_align(pred_v, gt_v)
    err_pa = np.linalg.norm(pred_pa - gt_v, axis=-1).mean(axis=-1)
    return {
        "valid_frames": int(np.count_nonzero(valid)),
        "joint_indices": joints.tolist(),
        "mpjpe_m": _summarize_errors(err),
        "root_aligned_mpjpe_m": _summarize_errors(err_ra),
        "pa_mpjpe_m": _summarize_errors(err_pa),
    }


def _point_track_error(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float | dict[str, float] | int]:
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape or not np.any(valid):
        return {"valid_frames": 0}
    diff = pred[valid] - gt[valid]
    return {
        "valid_frames": int(np.count_nonzero(valid)),
        "l2_m": _summarize_errors(np.linalg.norm(diff, axis=1)),
        "xyz_abs_m": {
            "x": _summarize_errors(np.abs(diff[:, 0])),
            "y": _summarize_errors(np.abs(diff[:, 1])),
            "z": _summarize_errors(np.abs(diff[:, 2])),
        },
    }


def _resolve_image_path(egobody_root: Path, image_path: object) -> Path:
    p = Path(str(image_path))
    if p.is_absolute():
        return p
    return egobody_root / p


def _world_to_camera(points_world: np.ndarray, T_world_pv: np.ndarray, camera_convention: str) -> np.ndarray:
    R = T_world_pv[:3, :3].astype(np.float32)
    t = T_world_pv[:3, 3].astype(np.float32)
    points_pv = (points_world.astype(np.float32) - t) @ R
    # export used points_pv = C @ points_cam. C is orthonormal/self-inverse for current modes.
    C = _camera_to_pv_matrix(camera_convention)
    points_cam = points_pv @ C.T
    return points_cam.astype(np.float32)


def _project(points_cam: np.ndarray, K: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    finite = np.isfinite(points_cam).all(axis=1) & (z > 1e-5)
    uv = np.full((len(points_cam), 2), np.nan, dtype=np.float32)
    uv[finite, 0] = K[0, 0] * points_cam[finite, 0] / z[finite] + K[0, 2]
    uv[finite, 1] = K[1, 1] * points_cam[finite, 1] / z[finite] + K[1, 2]
    in_image = finite & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, finite, in_image


def _draw_overlay(
    image_bgr: np.ndarray,
    vertices_world: np.ndarray,
    joints_world: np.ndarray,
    T_world_pv: np.ndarray,
    K: np.ndarray,
    camera_convention: str,
    point_stride: int,
) -> tuple[np.ndarray, dict[str, float]]:
    h, w = image_bgr.shape[:2]
    verts = vertices_world[:: max(1, point_stride)]
    verts_cam = _world_to_camera(verts, T_world_pv, camera_convention)
    verts_uv, verts_finite, verts_in = _project(verts_cam, K, w, h)

    joints_cam = _world_to_camera(joints_world, T_world_pv, camera_convention)
    joints_uv, joints_finite, joints_in = _project(joints_cam, K, w, h)

    overlay = image_bgr.copy()
    for u, v in verts_uv[verts_in].astype(np.int32):
        cv2.circle(overlay, (int(u), int(v)), 1, (255, 180, 20), -1, lineType=cv2.LINE_AA)
    for idx, (u, v) in enumerate(joints_uv):
        if not joints_in[idx]:
            continue
        color = (0, 0, 255) if idx != 0 else (0, 255, 255)
        radius = 3 if idx != 0 else 5
        cv2.circle(overlay, (int(u), int(v)), radius, color, -1, lineType=cv2.LINE_AA)

    alpha = 0.72
    image_bgr = cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0.0)
    stats = {
        "sampled_vertices": float(len(verts)),
        "vertex_positive_depth_fraction": float(np.mean(verts_finite)) if len(verts_finite) else 0.0,
        "vertex_in_image_fraction": float(np.mean(verts_in)) if len(verts_in) else 0.0,
        "joint_positive_depth_fraction": float(np.mean(joints_finite)) if len(joints_finite) else 0.0,
        "joint_in_image_fraction": float(np.mean(joints_in)) if len(joints_in) else 0.0,
    }
    if np.any(verts_in):
        xy = verts_uv[verts_in]
        stats.update(
            {
                "bbox_u_min": float(np.min(xy[:, 0])),
                "bbox_v_min": float(np.min(xy[:, 1])),
                "bbox_u_max": float(np.max(xy[:, 0])),
                "bbox_v_max": float(np.max(xy[:, 1])),
            }
        )
    return image_bgr, stats


def _summarize_errors(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {}
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "max": float(np.max(errors)),
    }


def _save_root_plot(
    out_path: Path,
    pred_root: np.ndarray,
    gt_root: np.ndarray | None,
    gt_valid: np.ndarray | None,
    pv_root: np.ndarray,
) -> None:
    t = np.arange(len(pred_root))
    names = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for axis, name in enumerate(names):
        ax = axes[axis]
        ax.plot(t, pred_root[:, axis], label="GVHMR exo exported", color="#a855f7", linewidth=1.8)
        ax.plot(t, pv_root[:, axis], label="PV camera", color="#0ea5e9", linewidth=1.0, alpha=0.75)
        if gt_root is not None and gt_valid is not None and np.any(gt_valid):
            ax.plot(t[gt_valid], gt_root[gt_valid, axis], label="interactee GT transl", color="#10b981", linewidth=1.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("output frame index")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exo-world-npz", type=Path, required=True, help="NPZ exported by export_gvhmr_exo_for_egoallo.py")
    parser.add_argument("--input-npz", type=Path, default=None, help="Converted EgoBody input NPZ. Defaults to path stored inside --exo-world-npz")
    parser.add_argument("--egobody-root", type=Path, default=Path("/public/home/wenxin/egobody"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--recording", default=None)
    parser.add_argument("--gt-split", default="auto", choices=["auto", "train", "val", "test"])
    parser.add_argument("--exo-gt-body-idx", type=int, default=None)
    parser.add_argument("--gt-coordinate-mode", default="kinect12_to_holo_zup", choices=["none", "kinect12_to_holo", "holo_y_up_to_z_up", "kinect12_to_holo_zup"])
    parser.add_argument("--camera-convention", default="auto", choices=["auto", "opencv_to_holo", "identity"])
    parser.add_argument("--num-overlays", type=int, default=6)
    parser.add_argument("--overlay-indices", type=int, nargs="*", default=None, help="Output-relative frame indices to render")
    parser.add_argument("--point-stride", type=int, default=12, help="Draw every Nth mesh vertex")
    parser.add_argument("--gt-smplx-model-path", type=Path, default=Path("/public/home/wenxin/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"))
    parser.add_argument("--gt-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gt-zero-betas", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compute-gt-joint-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute-exo-part-tracks", action=argparse.BooleanOptionalAction, default=False, help="Also report exo head/wrist joint track errors. Body MPJPE remains enabled by --compute-gt-joint-metrics.")
    parser.add_argument("--baseline-exo-world-npz", type=Path, default=None, help="Optional raw/baseline exo world NPZ for before-vs-after comparison")
    parser.add_argument("--ego-gt-body-idx", type=int, default=None, help="Camera wearer GT body_idx override for sensor trajectory diagnostics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    exo_path = args.exo_world_npz.resolve()
    exo = np.load(exo_path, allow_pickle=True)
    if args.input_npz is None:
        if "input_npz" not in exo.files:
            raise ValueError("--input-npz is required because exo NPZ has no input_npz field")
        input_path = Path(_np_scalar_to_str(exo["input_npz"])).resolve()
    else:
        input_path = args.input_npz.resolve()
    inp = np.load(input_path, allow_pickle=True)

    vertices_world = np.asarray(exo["vertices_world"], dtype=np.float32)
    joints_world = np.asarray(exo["joints_world"], dtype=np.float32)
    if vertices_world.ndim != 3 or vertices_world.shape[-1] != 3:
        raise ValueError(f"Bad vertices_world shape: {vertices_world.shape}")
    if joints_world.ndim != 3 or joints_world.shape[-1] != 3:
        raise ValueError(f"Bad joints_world shape: {joints_world.shape}")

    start = int(np.asarray(exo["start_index"]).item()) if "start_index" in exo.files else 0
    end = int(np.asarray(exo["end_index"]).item()) if "end_index" in exo.files else start + len(vertices_world)
    if end - start != len(vertices_world):
        raise ValueError(f"Interval [{start}, {end}) does not match exo length {len(vertices_world)}")

    camera_convention = args.camera_convention
    if camera_convention == "auto":
        camera_convention = _np_scalar_to_str(exo["camera_convention"]) if "camera_convention" in exo.files else "opencv_to_holo"

    T_world_pv = np.asarray(inp["Ts_world_pv_mats"], dtype=np.float32)[start:end]
    K_fullimg = np.asarray(inp["K_fullimg"], dtype=np.float32)[start:end]
    image_paths = np.asarray(inp["image_paths"], dtype=object)[start:end]
    if len(T_world_pv) != len(vertices_world):
        raise ValueError(f"PV pose length {len(T_world_pv)} does not match exo length {len(vertices_world)}")

    recording = args.recording
    if recording is None:
        recording = _np_scalar_to_str(inp["recording"]) if "recording" in inp.files else input_path.parent.name

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = exo_path.parent / f"{exo_path.stem}_reprojection_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    frame_ids = np.asarray([_parse_frame_id(p) if _parse_frame_id(p) is not None else -1 for p in image_paths], dtype=np.int64)
    gt_root, gt_valid = _load_gt_transl_sequence(
        args.egobody_root,
        recording,
        args.gt_split,
        "interactee",
        args.exo_gt_body_idx,
        frame_ids,
        args.gt_coordinate_mode,
    )

    exo_gt_joints = None
    exo_gt_joints_valid = None
    ego_gt_joints = None
    ego_gt_joints_valid = None
    if args.compute_gt_joint_metrics:
        exo_gt_joints, exo_gt_joints_valid = _load_gt_joints_sequence(
            args.egobody_root, recording, args.gt_split, "interactee", args.exo_gt_body_idx,
            frame_ids, args.gt_coordinate_mode, args.gt_smplx_model_path, args.gt_device, args.gt_zero_betas
        )
        ego_gt_joints, ego_gt_joints_valid = _load_gt_joints_sequence(
            args.egobody_root, recording, args.gt_split, "camera_wearer", args.ego_gt_body_idx,
            frame_ids, args.gt_coordinate_mode, args.gt_smplx_model_path, args.gt_device, args.gt_zero_betas
        )

    pred_root = joints_world[:, 0]
    pv_root = T_world_pv[:, :3, 3]
    metrics: dict[str, Any] = {
        "exo_world_npz": str(exo_path),
        "input_npz": str(input_path),
        "recording": recording,
        "start_index": start,
        "end_index": end,
        "num_frames": int(len(vertices_world)),
        "camera_convention": camera_convention,
        "gt_coordinate_mode": args.gt_coordinate_mode,
        "root_pred_min": np.min(pred_root, axis=0).tolist(),
        "root_pred_max": np.max(pred_root, axis=0).tolist(),
        "pv_camera_min": np.min(pv_root, axis=0).tolist(),
        "pv_camera_max": np.max(pv_root, axis=0).tolist(),
    }

    if len(gt_root) == len(pred_root) and np.any(gt_valid):
        pred_valid = pred_root[gt_valid]
        gt_valid_root = gt_root[gt_valid]
        diff = pred_valid - gt_valid_root
        root_l2 = np.linalg.norm(diff, axis=1)
        root_xy = np.linalg.norm(diff[:, :2], axis=1)
        root_z = np.abs(diff[:, 2])

        mean_offset = np.mean(diff, axis=0)
        first_offset = diff[0]
        mean_aligned_diff = diff - mean_offset[None]
        first_aligned_diff = diff - first_offset[None]

        metrics.update(
            {
                "gt_valid_frames": int(np.count_nonzero(gt_valid)),
                "root_offset_pred_minus_gt_mean_m": mean_offset.tolist(),
                "root_offset_pred_minus_gt_median_m": np.median(diff, axis=0).tolist(),
                "root_offset_pred_minus_gt_first_m": first_offset.tolist(),
                "root_l2_error_m": _summarize_errors(root_l2),
                "root_xy_error_m": _summarize_errors(root_xy),
                "root_z_error_m": _summarize_errors(root_z),
                "root_l2_error_mean_aligned_m": _summarize_errors(np.linalg.norm(mean_aligned_diff, axis=1)),
                "root_xy_error_mean_aligned_m": _summarize_errors(np.linalg.norm(mean_aligned_diff[:, :2], axis=1)),
                "root_z_error_mean_aligned_m": _summarize_errors(np.abs(mean_aligned_diff[:, 2])),
                "root_l2_error_first_aligned_m": _summarize_errors(np.linalg.norm(first_aligned_diff, axis=1)),
                "root_xy_error_first_aligned_m": _summarize_errors(np.linalg.norm(first_aligned_diff[:, :2], axis=1)),
                "root_z_error_first_aligned_m": _summarize_errors(np.abs(first_aligned_diff[:, 2])),
                "root_delta_pred_min_m": np.min(pred_valid - pred_valid[0], axis=0).tolist(),
                "root_delta_pred_max_m": np.max(pred_valid - pred_valid[0], axis=0).tolist(),
                "root_delta_gt_min_m": np.min(gt_valid_root - gt_valid_root[0], axis=0).tolist(),
                "root_delta_gt_max_m": np.max(gt_valid_root - gt_valid_root[0], axis=0).tolist(),
                "gt_root_min": np.min(gt_valid_root, axis=0).tolist(),
                "gt_root_max": np.max(gt_valid_root, axis=0).tolist(),
            }
        )
    else:
        metrics["gt_valid_frames"] = 0

    if exo_gt_joints is not None and exo_gt_joints_valid is not None and exo_gt_joints.ndim == 3:
        valid = exo_gt_joints_valid & np.isfinite(joints_world).all(axis=(1, 2)) & np.isfinite(exo_gt_joints).all(axis=(1, 2))
        metrics["exo_body_joint_metrics"] = _joint_metrics(joints_world, exo_gt_joints, valid, BODY22)
        if args.compute_exo_part_tracks and exo_gt_joints.shape[1] > RIGHT_WRIST_JOINT and joints_world.shape[1] > RIGHT_WRIST_JOINT:
            metrics["exo_head_track_error_m"] = _point_track_error(joints_world[:, HEAD_JOINT], exo_gt_joints[:, HEAD_JOINT], valid)
            metrics["exo_left_wrist_track_error_m"] = _point_track_error(joints_world[:, LEFT_WRIST_JOINT], exo_gt_joints[:, LEFT_WRIST_JOINT], valid)
            metrics["exo_right_wrist_track_error_m"] = _point_track_error(joints_world[:, RIGHT_WRIST_JOINT], exo_gt_joints[:, RIGHT_WRIST_JOINT], valid)

    if ego_gt_joints is not None and ego_gt_joints_valid is not None and ego_gt_joints.ndim == 3 and ego_gt_joints.shape[1] > RIGHT_WRIST_JOINT:
        sensor_valid = ego_gt_joints_valid & np.isfinite(ego_gt_joints).all(axis=(1, 2))
        if "Ts_world_head_mats" in inp.files:
            head_sensor = np.asarray(inp["Ts_world_head_mats"], dtype=np.float32)[start:end, :3, 3]
            metrics["ego_sensor_head_track_error_m"] = _point_track_error(head_sensor, ego_gt_joints[:, HEAD_JOINT], sensor_valid)
        elif "Ts_world_cpf_mats" in inp.files:
            cpf_sensor = np.asarray(inp["Ts_world_cpf_mats"], dtype=np.float32)[start:end, :3, 3]
            metrics["ego_sensor_cpf_to_gt_head_track_error_m"] = _point_track_error(cpf_sensor, ego_gt_joints[:, HEAD_JOINT], sensor_valid)
        for side, joint_id in [("left", LEFT_WRIST_JOINT), ("right", RIGHT_WRIST_JOINT)]:
            key = f"{side}_hand_wrist_position"
            available_key = f"{side}_hand_available"
            if key in inp.files:
                wrist_sensor = np.asarray(inp[key], dtype=np.float32)[start:end]
                valid = sensor_valid & np.isfinite(wrist_sensor).all(axis=1)
                if available_key in inp.files:
                    valid = valid & np.asarray(inp[available_key], dtype=bool)[start:end]
                metrics[f"ego_sensor_{side}_wrist_track_error_m"] = _point_track_error(wrist_sensor, ego_gt_joints[:, joint_id], valid)

    baseline_path = args.baseline_exo_world_npz
    if baseline_path is None and "joints_world_original" in exo.files:
        baseline_joints = np.asarray(exo["joints_world_original"], dtype=np.float32)
        baseline_vertices = np.asarray(exo["vertices_world_original"], dtype=np.float32) if "vertices_world_original" in exo.files else None
        baseline_source = "embedded_original"
    elif baseline_path is not None:
        baseline = np.load(baseline_path.resolve(), allow_pickle=True)
        baseline_joints = np.asarray(baseline["joints_world"], dtype=np.float32)
        baseline_vertices = np.asarray(baseline["vertices_world"], dtype=np.float32) if "vertices_world" in baseline.files else None
        baseline_source = str(baseline_path.resolve())
    else:
        baseline_joints = None
        baseline_vertices = None
        baseline_source = None

    if baseline_joints is not None and baseline_joints.shape == joints_world.shape:
        baseline_metrics: dict[str, Any] = {"source": baseline_source}
        if len(gt_root) == len(pred_root) and np.any(gt_valid):
            bdiff = baseline_joints[:, 0][gt_valid] - gt_root[gt_valid]
            baseline_metrics["root_l2_error_m"] = _summarize_errors(np.linalg.norm(bdiff, axis=1))
            baseline_metrics["root_z_error_m"] = _summarize_errors(np.abs(bdiff[:, 2]))
            if "root_l2_error_m" in metrics:
                baseline_metrics["delta_current_minus_baseline_root_l2_mean_m"] = (
                    metrics["root_l2_error_m"].get("mean", 0.0) - baseline_metrics["root_l2_error_m"].get("mean", 0.0)
                )
        if exo_gt_joints is not None and exo_gt_joints_valid is not None and exo_gt_joints.ndim == 3:
            valid = exo_gt_joints_valid & np.isfinite(baseline_joints).all(axis=(1, 2)) & np.isfinite(exo_gt_joints).all(axis=(1, 2))
            baseline_metrics["exo_body_joint_metrics"] = _joint_metrics(baseline_joints, exo_gt_joints, valid, BODY22)
            cur = metrics.get("exo_body_joint_metrics", {})
            base = baseline_metrics["exo_body_joint_metrics"]
            if "mpjpe_m" in cur and "mpjpe_m" in base:
                baseline_metrics["delta_current_minus_baseline_mpjpe_mean_m"] = cur["mpjpe_m"].get("mean", 0.0) - base["mpjpe_m"].get("mean", 0.0)
                baseline_metrics["delta_current_minus_baseline_pa_mpjpe_mean_m"] = cur["pa_mpjpe_m"].get("mean", 0.0) - base["pa_mpjpe_m"].get("mean", 0.0)
        metrics["baseline_comparison"] = baseline_metrics

    _save_root_plot(out_dir / "root_trajectory.png", pred_root, gt_root if len(gt_root) else None, gt_valid if len(gt_valid) else None, pv_root)

    if args.overlay_indices is not None and len(args.overlay_indices) > 0:
        overlay_indices = np.asarray(args.overlay_indices, dtype=np.int64)
    else:
        n = min(args.num_overlays, len(vertices_world))
        overlay_indices = np.linspace(0, len(vertices_world) - 1, n).round().astype(np.int64) if n > 0 else np.zeros((0,), dtype=np.int64)
    overlay_indices = overlay_indices[(overlay_indices >= 0) & (overlay_indices < len(vertices_world))]

    overlay_stats = []
    for rel_idx in overlay_indices:
        image_path = _resolve_image_path(args.egobody_root, image_paths[int(rel_idx)])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            overlay_stats.append({"rel_index": int(rel_idx), "image_path": str(image_path), "error": "failed_to_read_image"})
            continue
        overlay, stats = _draw_overlay(
            image,
            vertices_world[int(rel_idx)],
            joints_world[int(rel_idx)],
            T_world_pv[int(rel_idx)],
            K_fullimg[int(rel_idx)],
            camera_convention,
            args.point_stride,
        )
        cv2.putText(
            overlay,
            f"rel={int(rel_idx)} input={start + int(rel_idx)} frame={int(frame_ids[int(rel_idx)])}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 255, 255),
            2,
            cv2.LINE_AA,
        )
        out_img = overlay_dir / f"reprojection_{int(rel_idx):04d}_input_{start + int(rel_idx):06d}.jpg"
        cv2.imwrite(str(out_img), overlay)
        stats.update(
            {
                "rel_index": int(rel_idx),
                "input_index": int(start + rel_idx),
                "frame_id": int(frame_ids[int(rel_idx)]),
                "image_path": str(image_path),
                "output_path": str(out_img),
            }
        )
        overlay_stats.append(stats)

    metrics["overlays"] = overlay_stats
    if overlay_stats:
        for key in ["vertex_positive_depth_fraction", "vertex_in_image_fraction", "joint_positive_depth_fraction", "joint_in_image_fraction"]:
            vals = [s[key] for s in overlay_stats if key in s]
            if vals:
                metrics[f"overlay_{key}_mean"] = float(np.mean(vals))

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {metrics_path}")
    print(f"Saved {out_dir / 'root_trajectory.png'}")
    print(f"Saved overlays to {overlay_dir}")
    if "root_l2_error_m" in metrics:
        print("root_l2_error_m", metrics["root_l2_error_m"])
        print("root_z_error_m", metrics["root_z_error_m"])
    if "exo_body_joint_metrics" in metrics:
        print("exo_mpjpe_m", metrics["exo_body_joint_metrics"].get("mpjpe_m"))
        print("exo_pa_mpjpe_m", metrics["exo_body_joint_metrics"].get("pa_mpjpe_m"))
    for key in ["overlay_vertex_in_image_fraction_mean", "overlay_joint_in_image_fraction_mean"]:
        if key in metrics:
            print(key, metrics[key])


if __name__ == "__main__":
    main()
