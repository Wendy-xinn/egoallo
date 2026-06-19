#!/usr/bin/env python3
"""Convert one EgoBody recording to a custom EgoAllo inference input.

EgoAllo conditions on ``T_world_cpf``: the central pupil frame pose in a world
frame. EgoBody's ``head_hand_eye.csv`` gives a tracked HoloLens/head pose, not a
CPF pose. We therefore estimate a fixed ``T_head_cpf`` from gaze origins when
available, then compute ``T_world_cpf = T_world_head @ T_head_cpf``.

This script does not fake a Project Aria VRS/MPS directory. It writes a compact
NPZ consumed by ``egobody_inference.py`` in this folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


HEAD_HAND_EYE_COLS = 861
HAND_JOINT_COUNT = 26
HAND_PALM_INDEX = 0
HAND_WRIST_INDEX = 1
HAND_INDEX_METACARPAL_INDEX = 6
HAND_LITTLE_METACARPAL_INDEX = 21
HAND_JOINT_NAMES = [
    "Palm",
    "Wrist",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "ThumbTip",
    "IndexMetacarpal",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "IndexTip",
    "MiddleMetacarpal",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "MiddleTip",
    "RingMetacarpal",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "RingTip",
    "LittleMetacarpal",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
    "LittleTip",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_torch_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_img_timestamp(path_like: str) -> int:
    stem = Path(str(path_like)).stem
    return int(stem.split("_frame_")[0])


def _as_list(values: Iterable) -> list[str]:
    return [str(v) for v in list(values)]


def _find_head_hand_eye_csv(egobody_root: Path, recording: str) -> Path:
    rec_dir = egobody_root / "egocentric_gaze" / recording
    candidates = sorted(rec_dir.glob("*/*_head_hand_eye.csv"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No head_hand_eye CSV found under {rec_dir}")
    if len(candidates) > 1:
        print(f"[warn] Found multiple head_hand_eye CSV files for {recording}; using {candidates[0]}")
    return candidates[0]


def _read_head_hand_eye(csv_path: Path) -> dict[str, np.ndarray]:
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None]
    if raw.shape[1] < HEAD_HAND_EYE_COLS:
        raise ValueError(f"Expected at least {HEAD_HAND_EYE_COLS} columns in {csv_path}, got {raw.shape[1]}")

    head_mats = raw[:, 1:17].reshape(-1, 4, 4)

    left_valid_col = 17
    left_start = 18
    left_end = left_start + HAND_JOINT_COUNT * 16
    right_valid_col = left_end
    right_start = right_valid_col + 1
    right_end = right_start + HAND_JOINT_COUNT * 16

    left_hand_transs = raw[:, left_start:left_end].reshape(-1, HAND_JOINT_COUNT, 4, 4)[:, :, :3, 3]
    right_hand_transs = raw[:, right_start:right_end].reshape(-1, HAND_JOINT_COUNT, 4, 4)[:, :, :3, 3]
    left_hand_available = raw[:, left_valid_col] > 0.5
    right_hand_available = raw[:, right_valid_col] > 0.5

    gaze_origin = raw[:, 852:855]
    gaze_direction = raw[:, 856:859]
    gaze_distance = raw[:, 860:861]
    gaze_available = raw[:, 851] > 0.5

    finite = (
        np.isfinite(head_mats).all(axis=(1, 2))
        & np.isfinite(gaze_origin).all(axis=1)
        & np.isfinite(gaze_direction).all(axis=1)
    )
    left_hand_finite = np.isfinite(left_hand_transs).all(axis=(1, 2))
    right_hand_finite = np.isfinite(right_hand_transs).all(axis=(1, 2))
    return {
        "timestamps": raw[:, 0].astype(np.int64),
        "head_mats": head_mats,
        "left_hand_transs": left_hand_transs,
        "left_hand_available": left_hand_available & left_hand_finite,
        "right_hand_transs": right_hand_transs,
        "right_hand_available": right_hand_available & right_hand_finite,
        "gaze_origin": gaze_origin,
        "gaze_direction": gaze_direction,
        "gaze_distance": gaze_distance,
        "gaze_available": gaze_available & finite,
        "finite": finite,
    }


def _find_pv_txt(egobody_root: Path, recording: str) -> Path:
    rec_dir = egobody_root / "egocentric_color" / recording
    candidates = sorted(rec_dir.glob("*/*_pv.txt"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No PV txt found under {rec_dir}")
    if len(candidates) > 1:
        print(f"[warn] Found multiple PV txt files for {recording}; using {candidates[0]}")
    return candidates[0]


def _read_pv_txt(pv_txt: Path) -> tuple[dict[str, float], dict[int, dict[str, np.ndarray | float]]]:
    lines = [line.strip() for line in pv_txt.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"PV txt is too short: {pv_txt}")

    cx, cy, w, h = [float(x) for x in lines[0].split(",")]
    meta = {"cx": cx, "cy": cy, "w": w, "h": h}
    per_frame: dict[int, dict[str, np.ndarray | float]] = {}
    for line in lines[1:]:
        parts = line.split(",")
        ts = int(parts[0].strip())
        fx, fy = float(parts[1]), float(parts[2])
        vals = np.asarray([float(x) for x in parts[3:]], dtype=np.float64)
        if vals.size != 16:
            raise ValueError(f"Expected 16 pose values in {pv_txt}, got {vals.size}")
        per_frame[ts] = {"fx": fx, "fy": fy, "pv2world": vals.reshape(4, 4)}
    return meta, per_frame


def _nearest_indices(query_timestamps: np.ndarray, source_timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_timestamps.size == 0:
        raise ValueError("No source timestamps available for matching")

    order = np.argsort(source_timestamps)
    source_sorted = source_timestamps[order]
    idx_right = np.searchsorted(source_sorted, query_timestamps, side="left")
    idx_left = np.clip(idx_right - 1, 0, len(source_sorted) - 1)
    idx_right = np.clip(idx_right, 0, len(source_sorted) - 1)
    diff_left = np.abs(source_sorted[idx_left] - query_timestamps)
    diff_right = np.abs(source_sorted[idx_right] - query_timestamps)
    use_right = diff_right < diff_left
    sorted_idx = np.where(use_right, idx_right, idx_left)
    idx = order[sorted_idx]
    matched_ts = source_timestamps[idx]
    diff_ticks = np.minimum(diff_left, diff_right)
    return idx, matched_ts, diff_ticks


def _nearest_pv(
    query_timestamps: np.ndarray,
    pv_meta: dict[str, float],
    pv_frames: dict[int, dict[str, np.ndarray | float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pv_ts = np.asarray(sorted(pv_frames.keys()), dtype=np.int64)
    idx, matched_ts, diff_ticks = _nearest_indices(query_timestamps, pv_ts)

    mats = []
    K = []
    for ts in pv_ts[idx]:
        frame = pv_frames[int(ts)]
        mats.append(frame["pv2world"])
        K.append(
            np.asarray(
                [
                    [float(frame["fx"]), 0.0, pv_meta["cx"]],
                    [0.0, float(frame["fy"]), pv_meta["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )
    return np.stack(mats, axis=0), matched_ts, diff_ticks, np.stack(K, axis=0)


def _axis_conversion_matrix(mode: str) -> np.ndarray:
    if mode == "none":
        return np.eye(3, dtype=np.float64)
    if mode == "holo_y_up_to_z_up":
        # Proper rotation about +X: x'=x, y'=-z, z'=y. This maps HoloLens +Y up
        # into a z-up EgoAllo-style world without introducing a reflection.
        return np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown axis conversion mode: {mode}")


def _convert_c2w_pose_axes(
    mats: np.ndarray,
    axis_mode: str,
) -> np.ndarray:
    """Convert local-to-world poses from EgoBody/HoloLens to EgoAllo world.

    axis_mode changes the world basis only. The local CPF/head axes are left
    untouched because EgoBody's head-hand-eye poses are already expressed with a
    HoloLens/CPF-compatible device convention.
    """
    B = _axis_conversion_matrix(axis_mode)
    out = np.broadcast_to(np.eye(4, dtype=np.float64), mats.shape).copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", B, mats[:, :3, :3])
    out[:, :3, 3] = np.einsum("ij,nj->ni", B, mats[:, :3, 3])
    return out


def _make_c2w_pose_from_rt(
    rotations_holo: np.ndarray,
    translations_holo: np.ndarray,
    axis_mode: str,
) -> np.ndarray:
    mats = np.broadcast_to(np.eye(4, dtype=np.float64), (len(rotations_holo), 4, 4)).copy()
    mats[:, :3, :3] = rotations_holo
    mats[:, :3, 3] = translations_holo
    return _convert_c2w_pose_axes(mats, axis_mode)


def _head_cpf_rotation_matrix(mode: str) -> np.ndarray:
    if mode == "smpl":
        # EgoAllo builds CPF from the SMPL-H head joint. In the processed AMASS
        # data, CPF +X points body-left, +Y points up, and +Z points forward.
        # EgoBody/HoloLens head tracking uses the device convention X right,
        # Y up, Z backward, so this is a 180 degree rotation about local +Y.
        return np.diag([-1.0, 1.0, -1.0]).astype(np.float64)
    if mode == "identity":
        return np.eye(3, dtype=np.float64)
    raise ValueError(f"Unknown head CPF rotation mode: {mode}")


def _convert_points_axes(points: np.ndarray, mode: str) -> np.ndarray:
    B = _axis_conversion_matrix(mode)
    return np.einsum("ij,...j->...i", B, points)


def _safe_normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    valid = norm[..., 0] > 1e-8
    out = np.zeros_like(vec)
    out[valid] = vec[valid] / norm[valid]
    if fallback is not None:
        out[~valid] = fallback[~valid]
    return out


def _estimate_palm_normals(hand_joints: np.ndarray) -> np.ndarray:
    """Estimate palm normals from HoloLens/EgoBody 26-joint hand positions.

    Assumes the Windows hand-joint ordering used by EgoBody: palm=0, wrist=1,
    index metacarpal=6, little metacarpal=21. The sign is a convention; if
    guidance behaves badly, use egobody_inference.py --flip-hand-normal.
    """
    palm = hand_joints[:, HAND_PALM_INDEX]
    index_base = hand_joints[:, HAND_INDEX_METACARPAL_INDEX]
    little_base = hand_joints[:, HAND_LITTLE_METACARPAL_INDEX]
    normal = np.cross(index_base - palm, little_base - palm)
    fallback = np.zeros_like(normal)
    fallback[:, 2] = 1.0
    return _safe_normalize(normal, fallback=fallback)


def _hand_guidance_fields(hand_joints: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "wrist_position": hand_joints[:, HAND_WRIST_INDEX].astype(np.float32),
        "palm_position": hand_joints[:, HAND_PALM_INDEX].astype(np.float32),
        "palm_normal": _estimate_palm_normals(hand_joints).astype(np.float32),
    }


def _estimate_T_head_cpf(
    head_mats: np.ndarray,
    gaze_origin: np.ndarray,
    gaze_available: np.ndarray,
    mode: str,
    manual_translation: tuple[float, float, float],
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate a fixed head-to-CPF transform in EgoBody/HoloLens coordinates.

    EgoAllo's SMPL-H helper uses identity rotation from head to CPF and places
    CPF at the midpoint of the eyes. EgoBody does not expose eye mesh vertices at
    inference time, but its gaze origin is already in the same world frame as the
    head pose. We convert valid gaze origins into the head frame and take a
    robust median, giving a time-invariant T_head_cpf.
    """
    if mode == "head_origin":
        offset = np.zeros(3, dtype=np.float64)
        used = 0
        source = "head_origin"
    elif mode == "manual":
        offset = np.asarray(manual_translation, dtype=np.float64)
        used = 0
        source = "manual"
    elif mode == "gaze_median":
        R = head_mats[:, :3, :3]
        t = head_mats[:, :3, 3]
        rel_world = gaze_origin - t
        rel_head = np.einsum("nij,nj->ni", np.swapaxes(R, 1, 2), rel_world)
        valid = gaze_available & np.isfinite(rel_head).all(axis=1)
        if np.count_nonzero(valid) == 0:
            offset = np.asarray(manual_translation, dtype=np.float64)
            used = 0
            source = "manual_fallback_no_valid_gaze"
        else:
            offset = np.median(rel_head[valid], axis=0)
            used = int(np.count_nonzero(valid))
            source = "gaze_median"
    else:
        raise ValueError(f"Unknown cpf position mode: {mode}")

    T_head_cpf = np.eye(4, dtype=np.float64)
    T_head_cpf[:3, 3] = offset
    stats = {"source": source, "num_valid_gaze_frames": used, "translation_head_cpf": offset.tolist()}
    return T_head_cpf, stats


def _make_scene_points(traj_xyz: np.ndarray, floor_z: float, max_floor_points: int = 1600) -> np.ndarray:
    if traj_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    xy = traj_xyz[:, :2]
    lo = np.min(xy, axis=0) - 1.0
    hi = np.max(xy, axis=0) + 1.0
    side = max(2, int(np.sqrt(max_floor_points)))
    xs = np.linspace(lo[0], hi[0], side, dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], side, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    floor = np.stack([xx.reshape(-1), yy.reshape(-1), np.full(xx.size, floor_z, dtype=np.float32)], axis=-1)

    stride = max(1, traj_xyz.shape[0] // 256)
    path_points = traj_xyz[::stride].astype(np.float32)
    return np.concatenate([floor, path_points], axis=0)


def convert(args: argparse.Namespace) -> Path:
    egobody_root = args.egobody_root
    preprocess_path = egobody_root / "output" / "view1" / args.recording / "preprocess_view1.pt"
    if not preprocess_path.exists():
        raise FileNotFoundError(f"Missing preprocess file: {preprocess_path}")

    data = _load_torch_file(preprocess_path)
    if "imgname" not in data:
        raise KeyError(f"{preprocess_path} does not contain key 'imgname'")

    imgnames = _as_list(data["imgname"])
    if args.valid_only and "mask" in data and "valid" in data["mask"]:
        valid = np.asarray(data["mask"]["valid"], dtype=bool)
        imgnames = [p for p, keep in zip(imgnames, valid) if keep]

    if args.max_frames is not None:
        imgnames = imgnames[: args.max_frames]
    if len(imgnames) < args.min_frames:
        raise ValueError(f"Need at least {args.min_frames} frames, got {len(imgnames)}")

    query_ts = np.asarray([_extract_img_timestamp(p) for p in imgnames], dtype=np.int64)

    head_csv = _find_head_hand_eye_csv(egobody_root, args.recording)
    head_data = _read_head_hand_eye(head_csv)
    head_idx, matched_head_ts, head_diff_ticks = _nearest_indices(query_ts, head_data["timestamps"])
    head_mats_holo = head_data["head_mats"][head_idx]
    left_hand_joints_holo = head_data["left_hand_transs"][head_idx]
    right_hand_joints_holo = head_data["right_hand_transs"][head_idx]
    left_hand_available = head_data["left_hand_available"][head_idx]
    right_hand_available = head_data["right_hand_available"][head_idx]
    gaze_origin_holo = head_data["gaze_origin"][head_idx]
    gaze_direction_holo = head_data["gaze_direction"][head_idx]
    gaze_available = head_data["gaze_available"][head_idx]

    T_head_cpf_holo, cpf_stats = _estimate_T_head_cpf(
        head_mats_holo,
        gaze_origin_holo,
        gaze_available,
        args.cpf_position_mode,
        tuple(args.head_cpf_translation),
    )
    R_head_cpf_holo = _head_cpf_rotation_matrix(args.head_cpf_rotation)
    T_head_cpf_holo[:3, :3] = R_head_cpf_holo
    head_cpf_offset_holo = T_head_cpf_holo[:3, 3]
    cpf_trans_holo = (
        np.einsum("nij,j->ni", head_mats_holo[:, :3, :3], head_cpf_offset_holo)
        + head_mats_holo[:, :3, 3]
    )
    cpf_mats_holo = np.broadcast_to(
        np.eye(4, dtype=np.float64), (len(head_mats_holo), 4, 4)
    ).copy()
    cpf_mats_holo[:, :3, :3] = np.einsum("nij,jk->nik", head_mats_holo[:, :3, :3], R_head_cpf_holo)
    cpf_mats_holo[:, :3, 3] = cpf_trans_holo

    pv_txt = _find_pv_txt(egobody_root, args.recording)
    pv_meta, pv_frames = _read_pv_txt(pv_txt)
    pv_mats_holo, matched_pv_ts, pv_diff_ticks, K_fullimg = _nearest_pv(query_ts, pv_meta, pv_frames)

    if args.cpf_rotation_source == "head":
        cpf_rot_holo = np.einsum("nij,jk->nik", head_mats_holo[:, :3, :3], R_head_cpf_holo)
    elif args.cpf_rotation_source == "pv":
        cpf_rot_holo = np.einsum("nij,jk->nik", pv_mats_holo[:, :3, :3], R_head_cpf_holo)
    else:
        raise ValueError(f"Unknown cpf rotation source: {args.cpf_rotation_source}")

    Ts_world_cpf_mats = _make_c2w_pose_from_rt(
        cpf_rot_holo,
        cpf_trans_holo,
        args.axis_conversion,
    ).astype(np.float32)
    Ts_world_head_mats = _convert_c2w_pose_axes(
        head_mats_holo, args.axis_conversion
    ).astype(np.float32)
    Ts_world_pv_mats = _convert_c2w_pose_axes(
        pv_mats_holo, args.axis_conversion
    ).astype(np.float32)
    left_hand_joints_world = _convert_points_axes(left_hand_joints_holo, args.axis_conversion).astype(np.float32)
    right_hand_joints_world = _convert_points_axes(right_hand_joints_holo, args.axis_conversion).astype(np.float32)
    left_hand_fields = _hand_guidance_fields(left_hand_joints_world)
    right_hand_fields = _hand_guidance_fields(right_hand_joints_world)
    T_head_cpf_converted = T_head_cpf_holo.astype(np.float32)

    if np.max(head_diff_ticks) > args.max_time_diff_ticks:
        print(
            "[warn] max head timestamp mismatch is "
            f"{int(np.max(head_diff_ticks))} ticks ({np.max(head_diff_ticks) / 1e7:.3f}s)"
        )
    if np.max(pv_diff_ticks) > args.max_time_diff_ticks:
        print(
            "[warn] max PV timestamp mismatch is "
            f"{int(np.max(pv_diff_ticks))} ticks ({np.max(pv_diff_ticks) / 1e7:.3f}s)"
        )

    cpf_z = Ts_world_cpf_mats[:, 2, 3]
    floor_z = args.floor_z
    if floor_z is None:
        floor_z = float(np.median(cpf_z) - args.assumed_head_height_m)
    scene_points = _make_scene_points(Ts_world_cpf_mats[:, :3, 3], floor_z)

    rel_timestamps_sec = (query_ts - query_ts[0]).astype(np.float64) / 1e7
    timestamps_ns = np.round(rel_timestamps_sec * 1e9).astype(np.int64)

    out_dir = args.output_root / args.recording
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "egoallo_outputs").mkdir(exist_ok=True)
    out_path = out_dir / args.output_name

    np.savez_compressed(
        out_path,
        recording=args.recording,
        pose_source="head_gaze_estimated_cpf",
        cpf_position_mode=args.cpf_position_mode,
        cpf_rotation_source=args.cpf_rotation_source,
        Ts_world_cpf_mats=Ts_world_cpf_mats,
        Ts_world_device_mats=Ts_world_head_mats,
        Ts_world_head_mats=Ts_world_head_mats,
        Ts_world_pv_mats=Ts_world_pv_mats,
        Ts_holo_head_mats=head_mats_holo.astype(np.float32),
        Ts_holo_cpf_mats=cpf_mats_holo.astype(np.float32),
        Ts_holo_pv_mats=pv_mats_holo.astype(np.float32),
        T_head_cpf_holo=T_head_cpf_holo.astype(np.float32),
        T_head_cpf_converted=T_head_cpf_converted,
        hand_joint_names=np.asarray(HAND_JOINT_NAMES),
        hand_joint_convention="Windows.Mirage/EgoBody 26 joints: palm=0, wrist=1, index_metacarpal=6, little_metacarpal=21",
        left_hand_joints_holo=left_hand_joints_holo.astype(np.float32),
        right_hand_joints_holo=right_hand_joints_holo.astype(np.float32),
        left_hand_joints_world=left_hand_joints_world,
        right_hand_joints_world=right_hand_joints_world,
        left_hand_available=left_hand_available,
        right_hand_available=right_hand_available,
        left_hand_wrist_position=left_hand_fields["wrist_position"],
        left_hand_palm_position=left_hand_fields["palm_position"],
        left_hand_palm_normal=left_hand_fields["palm_normal"],
        right_hand_wrist_position=right_hand_fields["wrist_position"],
        right_hand_palm_position=right_hand_fields["palm_position"],
        right_hand_palm_normal=right_hand_fields["palm_normal"],
        gaze_origin_holo=gaze_origin_holo.astype(np.float32),
        gaze_direction_holo=gaze_direction_holo.astype(np.float32),
        gaze_available=gaze_available,
        K_fullimg=K_fullimg.astype(np.float32),
        query_timestamps=query_ts,
        matched_head_timestamps=matched_head_ts,
        matched_pv_timestamps=matched_pv_ts,
        head_timestamp_diff_ticks=head_diff_ticks,
        pv_timestamp_diff_ticks=pv_diff_ticks,
        timestamp_diff_ticks=head_diff_ticks,
        pose_timestamps_sec=rel_timestamps_sec.astype(np.float32),
        timestamps_ns=timestamps_ns,
        frame_nums=np.arange(len(imgnames), dtype=np.int64),
        image_paths=np.asarray(imgnames),
        floor_z=np.asarray(floor_z, dtype=np.float32),
        scene_points=scene_points,
        axis_conversion=args.axis_conversion,
        head_hand_eye_csv=str(head_csv),
        pv_txt=str(pv_txt),
        preprocess_path=str(preprocess_path),
    )

    metadata = {
        "recording": args.recording,
        "num_frames": len(imgnames),
        "pose_source": "head_gaze_estimated_cpf",
        "cpf_position_mode": args.cpf_position_mode,
        "cpf_rotation_source": args.cpf_rotation_source,
        "head_cpf_rotation": args.head_cpf_rotation,
        "cpf_stats": cpf_stats,
        "axis_conversion": args.axis_conversion,
        "coordinate_convention": "CPF position is estimated from head/gaze. CPF rotation uses a fixed head-to-CPF rotation before converting the world basis to EgoAllo z-up.",
        "floor_z": floor_z,
        "scene_points": int(scene_points.shape[0]),
        "head_hand_eye_csv": str(head_csv),
        "pv_txt": str(pv_txt),
        "preprocess_path": str(preprocess_path),
        "max_head_time_diff_ticks": int(np.max(head_diff_ticks)),
        "median_head_time_diff_ticks": float(np.median(head_diff_ticks)),
        "max_pv_time_diff_ticks": int(np.max(pv_diff_ticks)),
        "median_pv_time_diff_ticks": float(np.median(pv_diff_ticks)),
        "gaze_available_frames": int(np.count_nonzero(gaze_available)),
        "left_hand_available_frames": int(np.count_nonzero(left_hand_available)),
        "right_hand_available_frames": int(np.count_nonzero(right_hand_available)),
        "hand_joint_convention": "Windows hand joint order: palm=0, wrist=1, index_metacarpal=6, little_metacarpal=21",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "image_paths.txt").write_text("\n".join(imgnames) + "\n")
    np.save(out_dir / "scene_points.npy", scene_points)
    np.save(out_dir / "cpf_trajectory.npy", Ts_world_cpf_mats[:, :3, 3])
    np.save(out_dir / "head_trajectory.npy", Ts_world_head_mats[:, :3, 3])
    np.save(out_dir / "left_hand_wrist_trajectory.npy", left_hand_fields["wrist_position"])
    np.save(out_dir / "right_hand_wrist_trajectory.npy", right_hand_fields["wrist_position"])

    print(f"Saved {out_path}")
    print(json.dumps(metadata, indent=2))
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, help="EgoBody recording name")
    parser.add_argument("--egobody-root", type=Path, default=Path("/public/home/wenxin/egobody"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "egoallo_egobody_trajectories",
        help="Output directory for converted EgoBody trajectories.",
    )
    parser.add_argument(
        "--cpf-position-mode",
        choices=["gaze_median", "head_origin", "manual"],
        default="gaze_median",
        help="How to estimate the fixed T_head_cpf translation.",
    )
    parser.add_argument(
        "--head-cpf-translation",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Fallback/manual CPF translation in the EgoBody head frame, meters.",
    )
    parser.add_argument("--axis-conversion", choices=["holo_y_up_to_z_up", "none"], default="holo_y_up_to_z_up")
    parser.add_argument("--cpf-rotation-source", choices=["head", "pv"], default="head", help="Use HoloLens head tracking or PV camera rotation for T_world_cpf")
    parser.add_argument("--head-cpf-rotation", choices=["smpl", "identity"], default="smpl", help="Fixed local rotation from EgoBody/HoloLens head frame to EgoAllo/SMPL CPF frame")
    parser.add_argument("--output-name", default="egobody_egoallo_input.npz", help="Name of the converted NPZ written under the recording output directory")
    parser.add_argument("--floor-z", type=float, default=None, help="Override floor z in converted world coordinates")
    parser.add_argument("--assumed-head-height-m", type=float, default=1.6)
    parser.add_argument("--max-time-diff-ticks", type=int, default=2_000_000)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=129)
    parser.add_argument("--valid-only", action="store_true", help="Drop frames marked invalid in preprocess mask")
    return parser


def main() -> None:
    convert(build_parser().parse_args())


if __name__ == "__main__":
    main()
