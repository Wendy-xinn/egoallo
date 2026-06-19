#!/usr/bin/env python3
"""Postprocess/optimize EgoBody ego-exo NPZ outputs.

This first version is intentionally conservative: it optimizes only a per-frame
translation applied to the exported GVHMR exo mesh/joints. It does not change
body pose, shape, or hand pose.

Inputs:
  - input_npz: converted EgoBody/EgoAllo input, for floor_z and timestamps.
  - exo_world_npz: exported GVHMR exo world NPZ.
  - egoallo_npz: optional, recorded in metadata for later interaction losses.

Output:
  - an NPZ with the same visualization-facing keys as exo_world_npz:
    vertices_world, joints_world, faces.
  - extra debug keys: translation_delta, original_*.

This gives a stable hook for the next stage. Later, the same interface can be
replaced by a torch optimizer using reprojection, interaction, contact, and ego
constraints.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

OPENPOSE25_TO_SMPL24 = np.asarray([
    [1, 12],   # neck
    [2, 17],   # right shoulder
    [3, 19],   # right elbow
    [4, 21],   # right wrist
    [5, 16],   # left shoulder
    [6, 18],   # left elbow
    [7, 20],   # left wrist
    [8, 0],    # mid hip / pelvis
    [9, 2],    # right hip
    [10, 5],   # right knee
    [11, 8],   # right ankle
    [12, 1],   # left hip
    [13, 4],   # left knee
    [14, 7],   # left ankle
], dtype=np.int64)

CONTACT_JOINT_IDS = np.asarray([0, 1, 2, 12, 15, 16, 17, 18, 19, 20, 21], dtype=np.int64)
FOOT_JOINT_IDS = np.asarray([7, 10, 8, 11], dtype=np.int64)
EGO_HEAD_JOINT_ID = 15
EGO_HAND_JOINT_IDS = {"left": 20, "right": 21}
EXO_HEAD_JOINT_ID = 15
EXO_HAND_JOINT_IDS = {"left": 20, "right": 21}
EGO_COLLISION_JOINT_IDS = np.asarray([0, 3, 6, 9, 12, 15, 16, 17, 18, 19, 20, 21], dtype=np.int64)
EXO_COLLISION_JOINT_IDS = np.asarray([0, 3, 6, 9, 12, 15, 16, 17, 18, 19, 20, 21], dtype=np.int64)


def _scalar_to_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.asarray([1.0], dtype=np.float32)
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def _smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if sigma <= 0 or len(values) <= 2:
        return values.copy()
    kernel = _gaussian_kernel1d(sigma)
    pad = len(kernel) // 2
    if values.ndim == 1:
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)
    out = np.empty_like(values, dtype=np.float32)
    for dim in range(values.shape[1]):
        padded = np.pad(values[:, dim], (pad, pad), mode="edge")
        out[:, dim] = np.convolve(padded, kernel, mode="valid")
    return out


def _clip_vector_norm(vec: np.ndarray, max_norm: float | None) -> np.ndarray:
    if max_norm is None or max_norm <= 0:
        return vec
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norm, 1e-8))
    return (vec * scale).astype(np.float32)


def _resolve_floor_z(input_data: np.lib.npyio.NpzFile, override: float | None) -> float:
    if override is not None:
        return float(override)
    if "floor_z" in input_data.files:
        return float(np.asarray(input_data["floor_z"]).item())
    return 0.0




def _maybe_add_egoallo_src(path: Path) -> None:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_egoallo_smplh_joints(
    egoallo_npz: Path | None,
    smplh_npz_path: Path,
    egoallo_src: Path,
    device,
    dtype,
) -> tuple[Any | None, dict[str, Any]]:
    if egoallo_npz is None:
        return None, {"loaded": False, "reason": "no_egoallo_npz"}
    egoallo_npz = egoallo_npz.resolve()
    if not egoallo_npz.exists():
        return None, {"loaded": False, "reason": f"missing:{egoallo_npz}"}

    try:
        import torch

        _maybe_add_egoallo_src(egoallo_src)
        from egoallo import fncsmpl, fncsmpl_extensions
        from egoallo.transforms import SE3
    except Exception as exc:
        return None, {"loaded": False, "reason": f"import_failed:{type(exc).__name__}:{exc}"}

    data = np.load(egoallo_npz, allow_pickle=True)
    required = ["Ts_world_cpf", "body_quats", "left_hand_quats", "right_hand_quats", "betas"]
    missing = [key for key in required if key not in data.files]
    if missing:
        return None, {"loaded": False, "reason": f"missing_keys:{missing}"}

    body_quats = torch.as_tensor(data["body_quats"][:1], device=device, dtype=dtype)
    left_hand_quats = torch.as_tensor(data["left_hand_quats"][:1], device=device, dtype=dtype)
    right_hand_quats = torch.as_tensor(data["right_hand_quats"][:1], device=device, dtype=dtype)
    betas = torch.as_tensor(data["betas"][:1], device=device, dtype=dtype)
    Ts_world_cpf = torch.as_tensor(data["Ts_world_cpf"], device=device, dtype=dtype)

    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)
    shaped = body_model.with_shape(torch.mean(betas, dim=1, keepdim=True))
    fk_outputs = shaped.with_pose_decomposed(
        T_world_root=SE3.identity(device=device, dtype=dtype).parameters(),
        body_quats=body_quats,
        left_hand_quats=left_hand_quats,
        right_hand_quats=right_hand_quats,
    )
    T_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
        fk_outputs,
        Ts_world_cpf[None, ...],
    )
    fk_outputs = fk_outputs.with_new_T_world_root(T_world_root)
    joints = torch.cat(
        [T_world_root[..., None, 4:7], fk_outputs.Ts_world_joint[..., 4:7]],
        dim=-2,
    )[0]
    return joints, {
        "loaded": True,
        "path": str(egoallo_npz),
        "num_frames": int(joints.shape[0]),
        "num_joints": int(joints.shape[1]),
    }

def _camera_to_pv_matrix(mode: str) -> np.ndarray:
    if mode == "opencv_to_holo":
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    if mode == "identity":
        return np.eye(3, dtype=np.float32)
    raise ValueError(f"Unknown camera convention: {mode}")


def _parse_frame_id(path_like: object) -> int | None:
    stem = Path(str(path_like)).stem
    match = re.search(r"_frame_(\d+)$", stem)
    if match is None:
        match = re.search(r"frame_(\d+)$", stem)
    if match is None:
        return None
    return int(match.group(1))


def _find_keypoints_npz(input_data: np.lib.npyio.NpzFile, input_path: Path, egobody_root: Path | None, override: Path | None) -> Path | None:
    if override is not None:
        return override.resolve()
    if "image_paths" not in input_data.files or len(input_data["image_paths"]) == 0:
        return None
    image_path = Path(str(np.asarray(input_data["image_paths"], dtype=object)[0]))
    candidates: list[Path] = []
    if image_path.is_absolute():
        candidates.append(image_path.parent.parent / "keypoints.npz")
    else:
        if egobody_root is not None:
            candidates.append(egobody_root / image_path.parent.parent / "keypoints.npz")
        # Common layout when the converted NPZ lives under egoallo_egobody_trajectories/<recording>.
        candidates.append(input_path.parent / image_path.parent.parent / "keypoints.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_keypoints_for_window(
    input_data: np.lib.npyio.NpzFile,
    input_path: Path,
    start: int,
    end: int,
    egobody_root: Path | None,
    override: Path | None,
) -> tuple[np.ndarray | None, Path | None]:
    keypoints_path = _find_keypoints_npz(input_data, input_path, egobody_root, override)
    if keypoints_path is None or not keypoints_path.exists():
        return None, keypoints_path
    kp_data = np.load(keypoints_path, allow_pickle=True)
    if "keypoints" not in kp_data.files:
        raise KeyError(f"{keypoints_path} has no keypoints array")
    keypoints = np.asarray(kp_data["keypoints"], dtype=np.float32)
    if "imgname" not in kp_data.files or "image_paths" not in input_data.files:
        if end <= len(keypoints):
            return keypoints[start:end], keypoints_path
        return None, keypoints_path

    kp_names = [str(x) for x in np.asarray(kp_data["imgname"], dtype=object)]
    kp_by_name = {name: i for i, name in enumerate(kp_names)}
    image_paths = [str(x) for x in np.asarray(input_data["image_paths"], dtype=object)[start:end]]
    out = np.zeros((len(image_paths), keypoints.shape[1], keypoints.shape[2]), dtype=np.float32)
    matched = 0
    for i, name in enumerate(image_paths):
        idx = kp_by_name.get(name)
        if idx is None:
            # Fall back to matching by frame id because some paths can differ in prefix.
            frame_id = _parse_frame_id(name)
            if frame_id is not None:
                for candidate_name, candidate_idx in kp_by_name.items():
                    if _parse_frame_id(candidate_name) == frame_id:
                        idx = candidate_idx
                        break
        if idx is not None:
            out[i] = keypoints[idx]
            matched += 1
    if matched == 0:
        return None, keypoints_path
    return out, keypoints_path


def _resolve_camera_convention(exo: np.lib.npyio.NpzFile, override: str) -> str:
    if override != "auto":
        return override
    if "camera_convention" in exo.files:
        return _scalar_to_str(exo["camera_convention"])
    return "opencv_to_holo"


def _torch_device_from_arg(value: str):
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA for interaction optimizer, but CUDA is not available.")
    return torch.device(value)


def _project_world_to_image_torch(points_world, T_world_pv, K, camera_convention: str):
    import torch

    R = T_world_pv[:, :3, :3]
    t = T_world_pv[:, :3, 3]
    points_pv = torch.einsum("tnj,tjk->tnk", points_world - t[:, None, :], R)
    C = torch.as_tensor(_camera_to_pv_matrix(camera_convention), device=points_world.device, dtype=points_world.dtype)
    points_cam = torch.einsum("tnj,kj->tnk", points_pv, C)
    z = points_cam[..., 2].clamp_min(1e-4)
    uv = torch.stack(
        [
            K[:, None, 0, 0] * points_cam[..., 0] / z + K[:, None, 0, 2],
            K[:, None, 1, 1] * points_cam[..., 1] / z + K[:, None, 1, 2],
        ],
        dim=-1,
    )
    return uv, points_cam[..., 2]


def _apply_delta_yaw_torch(vertices, joints, delta, yaw):
    import torch

    root = joints[:, :1, :]
    cos = torch.cos(yaw)
    sin = torch.sin(yaw)
    zeros = torch.zeros_like(cos)
    ones = torch.ones_like(cos)
    R = torch.stack(
        [
            torch.stack([cos, -sin, zeros], dim=-1),
            torch.stack([sin, cos, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    vertices_out = torch.einsum("tnj,tjk->tnk", vertices - root, R.transpose(-1, -2)) + root + delta[:, None, :]
    joints_out = torch.einsum("tnj,tjk->tnk", joints - root, R.transpose(-1, -2)) + root + delta[:, None, :]
    return vertices_out, joints_out


def _accel_loss(values):
    if values.shape[0] < 3:
        return values.new_tensor(0.0)
    return ((values[2:] - 2.0 * values[1:-1] + values[:-2]) ** 2).mean()


def _velocity_loss(values):
    if values.shape[0] < 2:
        return values.new_tensor(0.0)
    return ((values[1:] - values[:-1]) ** 2).mean()


def _finite_np(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr).all(axis=-1)


def _run_interaction_v1(
    input_data: np.lib.npyio.NpzFile,
    input_path: Path,
    exo: np.lib.npyio.NpzFile,
    vertices: np.ndarray,
    joints: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    start = int(exo["start_index"]) if "start_index" in exo.files else 0
    end = int(exo["end_index"]) if "end_index" in exo.files else start + len(vertices)
    if end - start != len(vertices):
        end = start + len(vertices)
    if "Ts_world_pv_mats" not in input_data.files or "K_fullimg" not in input_data.files:
        raise KeyError("interaction_v1 requires Ts_world_pv_mats and K_fullimg in input NPZ")

    device = _torch_device_from_arg(args.interaction_device)
    dtype = torch.float32
    camera_convention = _resolve_camera_convention(exo, args.camera_convention)
    T_world_pv_np = np.asarray(input_data["Ts_world_pv_mats"], dtype=np.float32)[start:end]
    K_np = np.asarray(input_data["K_fullimg"], dtype=np.float32)[start:end]
    keypoints_np, keypoints_path = _load_keypoints_for_window(
        input_data,
        input_path,
        start,
        end,
        args.egobody_root,
        args.keypoints_npz,
    )

    vertices_t = torch.as_tensor(vertices, device=device, dtype=dtype)
    joints_t = torch.as_tensor(joints, device=device, dtype=dtype)
    T_world_pv = torch.as_tensor(T_world_pv_np, device=device, dtype=dtype)
    K = torch.as_tensor(K_np, device=device, dtype=dtype)
    ego_joints_t, ego_debug = _load_egoallo_smplh_joints(
        args.egoallo_npz,
        args.smplh_npz_path,
        args.egoallo_src,
        device,
        dtype,
    )
    if ego_joints_t is not None and ego_joints_t.shape[0] != len(vertices):
        min_len = min(int(ego_joints_t.shape[0]), len(vertices))
        ego_joints_t = ego_joints_t[:min_len]
        ego_debug["trimmed_to_frames"] = min_len
        if min_len != len(vertices):
            ego_debug["usable_for_losses"] = False
            ego_joints_t = None

    T = len(vertices)
    raw_delta = torch.zeros((T, 3), device=device, dtype=dtype, requires_grad=True)
    raw_yaw = torch.zeros((T,), device=device, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([raw_delta, raw_yaw], lr=args.interaction_lr)

    kp_obs = kp_conf = map_op = map_smpl = None
    reproj_available = False
    diag_px = torch.sqrt((2.0 * K[:, 0, 2]).clamp_min(1.0) ** 2 + (2.0 * K[:, 1, 2]).clamp_min(1.0) ** 2)
    if keypoints_np is not None:
        mapping = OPENPOSE25_TO_SMPL24.copy()
        mapping = mapping[(mapping[:, 0] < keypoints_np.shape[1]) & (mapping[:, 1] < joints.shape[1])]
        if len(mapping) > 0:
            kp = torch.as_tensor(keypoints_np, device=device, dtype=dtype)
            map_op = torch.as_tensor(mapping[:, 0], device=device, dtype=torch.long)
            map_smpl = torch.as_tensor(mapping[:, 1], device=device, dtype=torch.long)
            kp_obs = kp[:, map_op, :2]
            kp_conf = kp[:, map_op, 2]
            reproj_available = bool(torch.count_nonzero(kp_conf > args.keypoint_conf_thresh).item() > 0)

    hand_targets = []
    for side in ["left", "right"]:
        key = f"{side}_hand_wrist_position"
        available_key = f"{side}_hand_available"
        if key not in input_data.files:
            continue
        points = np.asarray(input_data[key], dtype=np.float32)[start:end]
        valid = _finite_np(points)
        if available_key in input_data.files:
            valid = valid & np.asarray(input_data[available_key], dtype=bool)[start:end]
        if np.count_nonzero(valid) == 0:
            continue
        hand_targets.append(
            (
                torch.as_tensor(points, device=device, dtype=dtype),
                torch.as_tensor(valid, device=device, dtype=torch.bool),
                side,
            )
        )

    sensor_head_targets = []
    if "Ts_world_head_mats" in input_data.files:
        head_points = np.asarray(input_data["Ts_world_head_mats"], dtype=np.float32)[start:end, :3, 3]
        head_valid = _finite_np(head_points)
        if np.count_nonzero(head_valid) > 0:
            sensor_head_targets.append(
                (
                    torch.as_tensor(head_points, device=device, dtype=dtype),
                    torch.as_tensor(head_valid, device=device, dtype=torch.bool),
                )
            )

    egoallo_hand_targets = []
    egoallo_head_target = None
    if ego_joints_t is not None and args.interaction_use_egoallo_hands:
        for side, joint_id in EGO_HAND_JOINT_IDS.items():
            if joint_id < ego_joints_t.shape[1]:
                valid = torch.isfinite(ego_joints_t[:, joint_id]).all(dim=-1)
                if torch.any(valid):
                    egoallo_hand_targets.append((ego_joints_t[:, joint_id], valid, side))
        if EGO_HEAD_JOINT_ID < ego_joints_t.shape[1]:
            valid = torch.isfinite(ego_joints_t[:, EGO_HEAD_JOINT_ID]).all(dim=-1)
            if torch.any(valid):
                egoallo_head_target = (ego_joints_t[:, EGO_HEAD_JOINT_ID], valid)

    ego_collision_ids_t = None
    exo_collision_ids_t = None
    if ego_joints_t is not None and args.interaction_collision_weight > 0:
        ego_collision_ids = EGO_COLLISION_JOINT_IDS[EGO_COLLISION_JOINT_IDS < ego_joints_t.shape[1]]
        exo_collision_ids = EXO_COLLISION_JOINT_IDS[EXO_COLLISION_JOINT_IDS < joints.shape[1]]
        if len(ego_collision_ids) > 0 and len(exo_collision_ids) > 0:
            ego_collision_ids_t = torch.as_tensor(ego_collision_ids, device=device, dtype=torch.long)
            exo_collision_ids_t = torch.as_tensor(exo_collision_ids, device=device, dtype=torch.long)

    sensor_collision_ids = EXO_COLLISION_JOINT_IDS[EXO_COLLISION_JOINT_IDS < joints.shape[1]]
    sensor_collision_ids_t = torch.as_tensor(sensor_collision_ids, device=device, dtype=torch.long)

    contact_joint_ids = CONTACT_JOINT_IDS[CONTACT_JOINT_IDS < joints.shape[1]]
    contact_joint_ids_t = torch.as_tensor(contact_joint_ids, device=device, dtype=torch.long)
    foot_joint_ids = FOOT_JOINT_IDS[FOOT_JOINT_IDS < joints.shape[1]]
    foot_joint_ids_t = torch.as_tensor(foot_joint_ids, device=device, dtype=torch.long)
    floor_z_value = _resolve_floor_z(input_data, args.floor_z)
    floor_z_t = torch.as_tensor(float(floor_z_value), device=device, dtype=dtype)
    foot_contact_mask = None
    if len(foot_joint_ids_t) > 0:
        with torch.no_grad():
            base_foot = joints_t[:, foot_joint_ids_t]
            base_vel = torch.linalg.norm(base_foot[1:] - base_foot[:-1], dim=-1)
            if base_vel.shape[0] > 0:
                base_vel = torch.cat([base_vel[:1], base_vel], dim=0)
            else:
                base_vel = torch.zeros(base_foot.shape[:2], device=device, dtype=dtype)
            base_height = torch.abs(base_foot[..., 2] - floor_z_t)
            foot_contact_mask = (base_vel < args.interaction_contact_vel_m) & (base_height < args.interaction_contact_height_m)
    max_xy = float(args.max_horizontal_correction_m)
    max_z = float(args.max_vertical_correction_m)
    max_yaw = np.deg2rad(float(args.interaction_max_yaw_deg))
    history: list[dict[str, float]] = []

    for it in range(args.interaction_iters):
        optimizer.zero_grad(set_to_none=True)
        delta_xy = max_xy * torch.tanh(raw_delta[:, :2])
        delta_z = max_z * torch.tanh(raw_delta[:, 2:3]) if args.interaction_optimize_z else torch.zeros_like(raw_delta[:, 2:3])
        delta = torch.cat([delta_xy, delta_z], dim=-1)
        yaw = max_yaw * torch.tanh(raw_yaw)
        _, joints_opt = _apply_delta_yaw_torch(vertices_t[:, :1], joints_t, delta, yaw)

        loss_terms: dict[str, torch.Tensor] = {}
        if reproj_available and args.interaction_reproj_weight > 0:
            pred_uv, pred_depth = _project_world_to_image_torch(joints_opt[:, map_smpl], T_world_pv, K, camera_convention)
            valid = (kp_conf > args.keypoint_conf_thresh) & torch.isfinite(kp_obs).all(dim=-1) & (pred_depth > 1e-4)
            if torch.any(valid):
                err = torch.linalg.norm((pred_uv - kp_obs) / diag_px[:, None, None].clamp_min(1.0), dim=-1)
                robust = torch.sqrt(err * err + args.interaction_reproj_huber ** 2) - args.interaction_reproj_huber
                weights = kp_conf.clamp_min(0.0)
                loss_terms["reproj"] = (robust[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-6)

        if hand_targets and args.interaction_hand_weight > 0 and len(contact_joint_ids_t) > 0:
            contact_points = joints_opt[:, contact_joint_ids_t]
            contact_losses = []
            contact_weights = []
            with torch.no_grad():
                base_contact = joints_t[:, contact_joint_ids_t]
            for hand_points, hand_valid, _side in hand_targets:
                dist = torch.linalg.norm(contact_points - hand_points[:, None, :], dim=-1).min(dim=-1).values
                base_dist = torch.linalg.norm(base_contact - hand_points[:, None, :], dim=-1).min(dim=-1).values
                activate = torch.sigmoid((args.interaction_hand_activation_m - base_dist) / max(args.interaction_hand_softness_m, 1e-4))
                valid = hand_valid & torch.isfinite(dist) & (activate > 1e-3)
                if torch.any(valid):
                    target = torch.clamp(dist[valid] - args.interaction_hand_target_m, min=0.0)
                    contact_losses.append(target * target)
                    contact_weights.append(activate[valid])
            if contact_losses:
                losses = torch.cat(contact_losses)
                weights = torch.cat(contact_weights)
                loss_terms["hand"] = (losses * weights).sum() / weights.sum().clamp_min(1e-6)

        if egoallo_hand_targets and args.interaction_egoexo_hand_weight > 0 and len(contact_joint_ids_t) > 0:
            contact_points = joints_opt[:, contact_joint_ids_t]
            ego_contact_losses = []
            ego_contact_weights = []
            with torch.no_grad():
                base_contact = joints_t[:, contact_joint_ids_t]
            for hand_points, hand_valid, _side in egoallo_hand_targets:
                dist = torch.linalg.norm(contact_points - hand_points[:, None, :], dim=-1).min(dim=-1).values
                base_dist = torch.linalg.norm(base_contact - hand_points[:, None, :], dim=-1).min(dim=-1).values
                activate = torch.sigmoid((args.interaction_egoexo_hand_activation_m - base_dist) / max(args.interaction_hand_softness_m, 1e-4))
                valid = hand_valid & torch.isfinite(dist) & (activate > 1e-3)
                if torch.any(valid):
                    target = torch.clamp(dist[valid] - args.interaction_egoexo_hand_target_m, min=0.0)
                    ego_contact_losses.append(target * target)
                    ego_contact_weights.append(activate[valid])
            if ego_contact_losses:
                losses = torch.cat(ego_contact_losses)
                weights = torch.cat(ego_contact_weights)
                loss_terms["egoexo_hand"] = (losses * weights).sum() / weights.sum().clamp_min(1e-6)

        if ego_joints_t is not None and ego_collision_ids_t is not None and exo_collision_ids_t is not None:
            if args.interaction_collision_weight > 0:
                exo_collision = joints_opt[:, exo_collision_ids_t]
                ego_collision = ego_joints_t[:, ego_collision_ids_t]
                dist = torch.cdist(exo_collision, ego_collision)
                penetration = torch.clamp(args.interaction_collision_margin_m - dist, min=0.0)
                loss_terms["egoexo_collision"] = (penetration * penetration).mean()

        head_collision_losses = []
        if len(sensor_collision_ids_t) > 0 and args.interaction_head_collision_weight > 0:
            exo_collision = joints_opt[:, sensor_collision_ids_t]
            for head_points, head_valid in sensor_head_targets:
                dist = torch.linalg.norm(exo_collision - head_points[:, None, :], dim=-1).min(dim=-1).values
                valid = head_valid & torch.isfinite(dist)
                if torch.any(valid):
                    penetration = torch.clamp(args.interaction_head_collision_margin_m - dist[valid], min=0.0)
                    head_collision_losses.append(penetration * penetration)
            if egoallo_head_target is not None:
                head_points, head_valid = egoallo_head_target
                dist = torch.linalg.norm(exo_collision - head_points[:, None, :], dim=-1).min(dim=-1).values
                valid = head_valid & torch.isfinite(dist)
                if torch.any(valid):
                    penetration = torch.clamp(args.interaction_head_collision_margin_m - dist[valid], min=0.0)
                    head_collision_losses.append(penetration * penetration)
            if head_collision_losses:
                loss_terms["head_collision"] = torch.cat(head_collision_losses).mean()

        if len(foot_joint_ids_t) > 0:
            foot_points = joints_opt[:, foot_joint_ids_t]
            foot_z = foot_points[..., 2]
            if args.interaction_floor_weight > 0:
                penetration = torch.clamp(floor_z_t + args.interaction_floor_margin_m - foot_z, min=0.0)
                loss_terms["floor"] = (penetration * penetration).mean()
            if foot_contact_mask is not None and args.interaction_contact_weight > 0 and torch.any(foot_contact_mask):
                contact_height = foot_z[foot_contact_mask] - floor_z_t
                loss_terms["contact_height"] = (contact_height * contact_height).mean()
            if foot_contact_mask is not None and args.interaction_skate_weight > 0 and foot_points.shape[0] > 1:
                pair_contact = foot_contact_mask[1:] & foot_contact_mask[:-1]
                if torch.any(pair_contact):
                    foot_xy_vel = foot_points[1:, :, :2] - foot_points[:-1, :, :2]
                    loss_terms["skate"] = (foot_xy_vel[pair_contact] * foot_xy_vel[pair_contact]).mean()

        root_opt = joints_opt[:, 0]
        if args.interaction_root_accel_weight > 0:
            loss_terms["root_accel"] = _accel_loss(root_opt)
        if args.interaction_root_z_accel_weight > 0:
            loss_terms["root_z_accel"] = _accel_loss(root_opt[:, 2:3])
        if args.interaction_delta_vel_weight > 0:
            loss_terms["delta_vel"] = _velocity_loss(delta)

        if args.interaction_anchor_weight > 0:
            loss_terms["anchor"] = (delta * delta).mean()
        if args.interaction_yaw_anchor_weight > 0:
            loss_terms["yaw_anchor"] = (yaw * yaw).mean()
        if args.interaction_smooth_weight > 0:
            loss_terms["smooth"] = _accel_loss(delta)
        if args.interaction_yaw_smooth_weight > 0:
            loss_terms["yaw_smooth"] = _accel_loss(yaw[:, None])

        total = delta.new_tensor(0.0)
        total = total + args.interaction_reproj_weight * loss_terms.get("reproj", total.new_tensor(0.0))
        total = total + args.interaction_hand_weight * loss_terms.get("hand", total.new_tensor(0.0))
        total = total + args.interaction_egoexo_hand_weight * loss_terms.get("egoexo_hand", total.new_tensor(0.0))
        total = total + args.interaction_collision_weight * loss_terms.get("egoexo_collision", total.new_tensor(0.0))
        total = total + args.interaction_head_collision_weight * loss_terms.get("head_collision", total.new_tensor(0.0))
        total = total + args.interaction_floor_weight * loss_terms.get("floor", total.new_tensor(0.0))
        total = total + args.interaction_contact_weight * loss_terms.get("contact_height", total.new_tensor(0.0))
        total = total + args.interaction_skate_weight * loss_terms.get("skate", total.new_tensor(0.0))
        total = total + args.interaction_root_accel_weight * loss_terms.get("root_accel", total.new_tensor(0.0))
        total = total + args.interaction_root_z_accel_weight * loss_terms.get("root_z_accel", total.new_tensor(0.0))
        total = total + args.interaction_delta_vel_weight * loss_terms.get("delta_vel", total.new_tensor(0.0))
        total = total + args.interaction_anchor_weight * loss_terms.get("anchor", total.new_tensor(0.0))
        total = total + args.interaction_yaw_anchor_weight * loss_terms.get("yaw_anchor", total.new_tensor(0.0))
        total = total + args.interaction_smooth_weight * loss_terms.get("smooth", total.new_tensor(0.0))
        total = total + args.interaction_yaw_smooth_weight * loss_terms.get("yaw_smooth", total.new_tensor(0.0))
        total.backward()
        optimizer.step()

        if it == 0 or it == args.interaction_iters - 1 or (it + 1) % max(1, args.interaction_log_every) == 0:
            row = {"iter": float(it), "total": float(total.detach().cpu())}
            for name, value in loss_terms.items():
                row[name] = float(value.detach().cpu())
            history.append(row)

    with torch.no_grad():
        delta_xy = max_xy * torch.tanh(raw_delta[:, :2])
        delta_z = max_z * torch.tanh(raw_delta[:, 2:3]) if args.interaction_optimize_z else torch.zeros_like(raw_delta[:, 2:3])
        delta = torch.cat([delta_xy, delta_z], dim=-1)
        yaw = max_yaw * torch.tanh(raw_yaw)
        vertices_opt, joints_opt = _apply_delta_yaw_torch(vertices_t, joints_t, delta, yaw)

    debug = {
        "method": "interaction_v1",
        "start_index": start,
        "end_index": end,
        "device": str(device),
        "camera_convention": camera_convention,
        "keypoints_npz": str(keypoints_path) if keypoints_path is not None else None,
        "keypoints_loaded": keypoints_np is not None,
        "keypoint_conf_thresh": args.keypoint_conf_thresh,
        "num_hand_target_sides": len(hand_targets),
        "num_sensor_head_targets": len(sensor_head_targets),
        "egoallo_joints": ego_debug,
        "num_egoallo_hand_target_sides": len(egoallo_hand_targets),
        "egoallo_head_target": egoallo_head_target is not None,
        "collision_enabled": ego_collision_ids_t is not None and exo_collision_ids_t is not None and args.interaction_collision_weight > 0,
        "interaction_iters": args.interaction_iters,
        "interaction_lr": args.interaction_lr,
        "weights": {
            "reproj": args.interaction_reproj_weight,
            "hand": args.interaction_hand_weight,
            "egoexo_hand": args.interaction_egoexo_hand_weight,
            "egoexo_collision": args.interaction_collision_weight,
            "head_collision": args.interaction_head_collision_weight,
            "floor": args.interaction_floor_weight,
            "contact_height": args.interaction_contact_weight,
            "skate": args.interaction_skate_weight,
            "root_accel": args.interaction_root_accel_weight,
            "root_z_accel": args.interaction_root_z_accel_weight,
            "delta_vel": args.interaction_delta_vel_weight,
            "anchor": args.interaction_anchor_weight,
            "yaw_anchor": args.interaction_yaw_anchor_weight,
            "smooth": args.interaction_smooth_weight,
            "yaw_smooth": args.interaction_yaw_smooth_weight,
        },
        "floor_z": float(floor_z_value),
        "foot_joint_ids": foot_joint_ids.tolist(),
        "num_foot_contact_entries": int(torch.count_nonzero(foot_contact_mask).detach().cpu()) if foot_contact_mask is not None else 0,
        "interaction_contact_height_m": args.interaction_contact_height_m,
        "interaction_contact_vel_m": args.interaction_contact_vel_m,
        "loss_history": history,
        "delta_min": delta.detach().cpu().numpy().min(axis=0).tolist(),
        "delta_median": np.median(delta.detach().cpu().numpy(), axis=0).tolist(),
        "delta_max": delta.detach().cpu().numpy().max(axis=0).tolist(),
        "yaw_delta_deg_min": float(np.rad2deg(yaw.detach().cpu().numpy()).min()),
        "yaw_delta_deg_median": float(np.median(np.rad2deg(yaw.detach().cpu().numpy()))),
        "yaw_delta_deg_max": float(np.rad2deg(yaw.detach().cpu().numpy()).max()),
    }
    return (
        vertices_opt.detach().cpu().numpy().astype(np.float32),
        joints_opt.detach().cpu().numpy().astype(np.float32),
        delta.detach().cpu().numpy().astype(np.float32),
        debug,
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--exo-world-npz", type=Path, required=True)
    parser.add_argument("--egoallo-npz", type=Path, default=None)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--method", choices=["translation_smooth_ground", "interaction_v1", "copy"], default="interaction_v1")
    parser.add_argument("--root-smooth-sigma", type=float, default=2.0)
    parser.add_argument("--root-smooth-weight", type=float, default=1.0)
    parser.add_argument("--smooth-z", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-horizontal-correction-m", type=float, default=0.08)
    parser.add_argument("--max-vertical-correction-m", type=float, default=0.06)
    parser.add_argument("--ground-weight", type=float, default=0.0)
    parser.add_argument("--ground-quantile", type=float, default=0.01, help="Vertex z quantile used as foot/floor proxy")
    parser.add_argument("--ground-smooth-sigma", type=float, default=4.0)
    parser.add_argument("--floor-z", type=float, default=None)
    parser.add_argument("--egobody-root", type=Path, default=Path("/public/home/wenxin/egobody"))
    parser.add_argument("--keypoints-npz", type=Path, default=None)
    parser.add_argument("--camera-convention", choices=["auto", "opencv_to_holo", "identity"], default="opencv_to_holo")
    parser.add_argument("--interaction-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--interaction-iters", type=int, default=180)
    parser.add_argument("--interaction-lr", type=float, default=0.05)
    parser.add_argument("--interaction-log-every", type=int, default=30)
    parser.add_argument("--interaction-optimize-z", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-max-yaw-deg", type=float, default=8.0)
    parser.add_argument("--keypoint-conf-thresh", type=float, default=0.25)
    parser.add_argument("--interaction-reproj-weight", type=float, default=0.2)
    parser.add_argument("--interaction-reproj-huber", type=float, default=0.02, help="Pseudo-Huber delta after normalizing pixel error by image diagonal")
    parser.add_argument("--interaction-hand-weight", type=float, default=0.25)
    parser.add_argument("--interaction-hand-activation-m", type=float, default=0.45)
    parser.add_argument("--interaction-hand-target-m", type=float, default=0.12)
    parser.add_argument("--interaction-hand-softness-m", type=float, default=0.08)
    parser.add_argument("--smplh-npz-path", type=Path, default=Path("/public/home/wenxin/egoallo/data/smplh/neutral/model.npz"))
    parser.add_argument("--egoallo-src", type=Path, default=Path("/public/home/wenxin/egoallo/src"))
    parser.add_argument("--interaction-use-egoallo-hands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-egoexo-hand-weight", type=float, default=0.05)
    parser.add_argument("--interaction-egoexo-hand-activation-m", type=float, default=0.45)
    parser.add_argument("--interaction-egoexo-hand-target-m", type=float, default=0.12)
    parser.add_argument("--interaction-collision-weight", type=float, default=0.05)
    parser.add_argument("--interaction-collision-margin-m", type=float, default=0.08)
    parser.add_argument("--interaction-head-collision-weight", type=float, default=0.05)
    parser.add_argument("--interaction-head-collision-margin-m", type=float, default=0.18)
    parser.add_argument("--interaction-floor-weight", type=float, default=0.2)
    parser.add_argument("--interaction-floor-margin-m", type=float, default=0.0)
    parser.add_argument("--interaction-contact-weight", type=float, default=0.5)
    parser.add_argument("--interaction-skate-weight", type=float, default=2.0)
    parser.add_argument("--interaction-contact-height-m", type=float, default=0.18)
    parser.add_argument("--interaction-contact-vel-m", type=float, default=0.025)
    parser.add_argument("--interaction-root-accel-weight", type=float, default=1.0)
    parser.add_argument("--interaction-root-z-accel-weight", type=float, default=3.0)
    parser.add_argument("--interaction-delta-vel-weight", type=float, default=1.0)
    parser.add_argument("--interaction-anchor-weight", type=float, default=0.5)
    parser.add_argument("--interaction-yaw-anchor-weight", type=float, default=0.02)
    parser.add_argument("--interaction-smooth-weight", type=float, default=8.0)
    parser.add_argument("--interaction-yaw-smooth-weight", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input_npz.resolve()
    exo_path = args.exo_world_npz.resolve()
    out_path = args.output_npz.resolve()
    egoallo_path = args.egoallo_npz.resolve() if args.egoallo_npz is not None else None

    input_data = np.load(input_path, allow_pickle=True)
    exo = np.load(exo_path, allow_pickle=True)
    required = ["vertices_world", "joints_world", "faces"]
    missing = [key for key in required if key not in exo.files]
    if missing:
        raise KeyError(f"{exo_path} is missing {missing}")

    vertices = np.asarray(exo["vertices_world"], dtype=np.float32)
    joints = np.asarray(exo["joints_world"], dtype=np.float32)
    faces = np.asarray(exo["faces"], dtype=np.int32)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Bad vertices_world shape: {vertices.shape}")
    if joints.ndim != 3 or joints.shape[-1] != 3 or len(joints) != len(vertices):
        raise ValueError(f"Bad joints_world shape: {joints.shape}")

    root = joints[:, 0].astype(np.float32)
    delta = np.zeros_like(root, dtype=np.float32)
    floor_z = _resolve_floor_z(input_data, args.floor_z)
    debug: dict[str, Any] = {
        "method": args.method,
        "floor_z": floor_z,
        "root_smooth_sigma": args.root_smooth_sigma,
        "root_smooth_weight": args.root_smooth_weight,
        "smooth_z": args.smooth_z,
        "ground_weight": args.ground_weight,
        "ground_quantile": args.ground_quantile,
        "ground_smooth_sigma": args.ground_smooth_sigma,
    }

    if args.method == "translation_smooth_ground":
        smooth_root = _smooth_1d(root, args.root_smooth_sigma)
        smooth_delta = (smooth_root - root) * float(args.root_smooth_weight)
        if not args.smooth_z:
            smooth_delta[:, 2] = 0.0
        smooth_delta[:, :2] = _clip_vector_norm(smooth_delta[:, :2], args.max_horizontal_correction_m)
        if args.smooth_z:
            smooth_delta[:, 2:3] = np.clip(
                smooth_delta[:, 2:3],
                -float(args.max_vertical_correction_m),
                float(args.max_vertical_correction_m),
            )
        delta += smooth_delta.astype(np.float32)

        q = float(np.clip(args.ground_quantile, 0.0, 0.2))
        foot_z = np.quantile(vertices[:, :, 2], q, axis=1).astype(np.float32)
        if args.ground_weight > 0:
            ground_delta_z = (floor_z - foot_z) * float(args.ground_weight)
            ground_delta_z = _smooth_1d(ground_delta_z, args.ground_smooth_sigma)
            ground_delta_z = np.clip(
                ground_delta_z,
                -float(args.max_vertical_correction_m),
                float(args.max_vertical_correction_m),
            )
            delta[:, 2] += ground_delta_z.astype(np.float32)
        else:
            ground_delta_z = np.zeros_like(foot_z, dtype=np.float32)
        debug.update(
            {
                "foot_z_before_min": float(np.min(foot_z)),
                "foot_z_before_median": float(np.median(foot_z)),
                "foot_z_before_max": float(np.max(foot_z)),
                "ground_delta_z_min": float(np.min(ground_delta_z)),
                "ground_delta_z_median": float(np.median(ground_delta_z)),
                "ground_delta_z_max": float(np.max(ground_delta_z)),
            }
        )
        vertices_opt = vertices + delta[:, None, :]
        joints_opt = joints + delta[:, None, :]
    elif args.method == "interaction_v1":
        vertices_opt, joints_opt, delta, interaction_debug = _run_interaction_v1(
            input_data,
            input_path,
            exo,
            vertices,
            joints,
            args,
        )
        debug.update(interaction_debug)
    elif args.method == "copy":
        vertices_opt = vertices.copy()
        joints_opt = joints.copy()
    else:
        raise ValueError(args.method)

    root_after = joints_opt[:, 0]
    debug.update(
        {
            "num_frames": int(len(vertices)),
            "delta_min": np.min(delta, axis=0).tolist(),
            "delta_median": np.median(delta, axis=0).tolist(),
            "delta_max": np.max(delta, axis=0).tolist(),
            "root_before_min": np.min(root, axis=0).tolist(),
            "root_before_max": np.max(root, axis=0).tolist(),
            "root_after_min": np.min(root_after, axis=0).tolist(),
            "root_after_max": np.max(root_after, axis=0).tolist(),
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {
        "vertices_world": vertices_opt.astype(np.float32),
        "joints_world": joints_opt.astype(np.float32),
        "faces": faces,
        "translation_delta": delta.astype(np.float32),
        "vertices_world_original": vertices.astype(np.float32),
        "joints_world_original": joints.astype(np.float32),
        "input_npz": str(input_path),
        "exo_world_npz": str(exo_path),
        "optimization_method": args.method,
        "optimization_debug_json": json.dumps(debug, indent=2),
    }
    for key in ["gvhmr_results", "start_index", "end_index", "camera_convention"]:
        if key in exo.files:
            save_kwargs[key] = exo[key]
    if args.method == "interaction_v1" and "yaw_delta_deg_min" in debug:
        save_kwargs["interaction_debug_json"] = json.dumps(debug, indent=2)
    if egoallo_path is not None:
        save_kwargs["egoallo_npz"] = str(egoallo_path)
    np.savez_compressed(out_path, **save_kwargs)

    print(f"Saved {out_path}")
    print(json.dumps(debug, indent=2))


if __name__ == "__main__":
    main()
