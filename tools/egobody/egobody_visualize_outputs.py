#!/usr/bin/env python3
"""Visualize EgoAllo outputs produced from EgoBody-converted trajectories.

This bypasses EgoAllo's original Aria/VRS discovery path. It directly loads the
NPZ saved by tools/egobody/egobody_inference.py and reuses EgoAllo's mesh
visualizer with no VRS video, MPS trajectory, HaMeR detections, or Aria hand
tracking requirements.

For EgoBody debugging, it can also overlay the camera wearer's and interactee's
SMPL-X GT meshes. The GT overlays are intentionally diagnostic: they keep
SMPL-X and SMPL-H as separate meshes instead of forcing an early SMPL-X ->
SMPL-H conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import viser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _maybe_add_egoallo_src(path: Path) -> None:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(value)


def _np_scalar_to_str(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def _source_input_path(npz_path: Path, outputs: np.lib.npyio.NpzFile) -> Path | None:
    candidates: list[Path] = []
    if "source_input" in outputs.files:
        source = _np_scalar_to_str(outputs["source_input"])
        if source and source != "None":
            candidates.append(Path(source))
    candidates.append(npz_path.resolve().parent.parent / "egobody_egoallo_input.npz")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_input_metadata(
    npz_path: Path,
    outputs: np.lib.npyio.NpzFile,
) -> tuple[np.lib.npyio.NpzFile | None, Path | None]:
    source_path = _source_input_path(npz_path, outputs)
    if source_path is None:
        return None, None
    return np.load(source_path, allow_pickle=True), source_path


def _resolve_floor_z(
    npz_path: Path,
    outputs: np.lib.npyio.NpzFile,
    input_data: np.lib.npyio.NpzFile | None,
    floor_z_override: float | None,
) -> float:
    if floor_z_override is not None:
        return float(floor_z_override)
    if "floor_z" in outputs.files:
        return float(outputs["floor_z"])
    if input_data is not None and "floor_z" in input_data.files:
        return float(input_data["floor_z"])
    if "Ts_world_cpf" in outputs.files:
        return float(np.median(outputs["Ts_world_cpf"][..., 6]) - 1.6)
    return 0.0


def _load_scene_points(
    input_data: np.lib.npyio.NpzFile | None,
    max_scene_points: int,
) -> np.ndarray | None:
    if input_data is None or "scene_points" not in input_data.files:
        return None
    points = np.asarray(input_data["scene_points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return None
    if max_scene_points > 0 and len(points) > max_scene_points:
        indices = np.linspace(0, len(points) - 1, max_scene_points).astype(np.int64)
        points = points[indices]
    return points


def _build_traj(outputs: np.lib.npyio.NpzFile, device: torch.device):
    from egoallo.network import EgoDenoiseTraj
    from egoallo.transforms import SO3

    (num_samples, timesteps, _, _) = outputs["body_quats"].shape
    body_quats = torch.from_numpy(outputs["body_quats"]).to(
        device=device, dtype=torch.float32
    )
    left_hand_quats = torch.from_numpy(outputs["left_hand_quats"]).to(
        device=device, dtype=torch.float32
    )
    right_hand_quats = torch.from_numpy(outputs["right_hand_quats"]).to(
        device=device, dtype=torch.float32
    )
    hand_quats = torch.cat([left_hand_quats, right_hand_quats], dim=-2)

    if "contacts" in outputs.files:
        contacts = torch.from_numpy(outputs["contacts"]).to(
            device=device, dtype=torch.float32
        )
    else:
        contacts = torch.zeros((num_samples, timesteps, 21), device=device)

    return EgoDenoiseTraj(
        betas=torch.from_numpy(outputs["betas"]).to(device=device, dtype=torch.float32),
        body_rotmats=SO3(body_quats).as_matrix(),
        contacts=contacts,
        hand_rotmats=SO3(hand_quats).as_matrix(),
    )


def _check_output_keys(outputs: np.lib.npyio.NpzFile) -> None:
    expected = [
        "Ts_world_cpf",
        "Ts_world_root",
        "body_quats",
        "left_hand_quats",
        "right_hand_quats",
        "betas",
        "frame_nums",
        "timestamps_ns",
    ]
    missing = [key for key in expected if key not in outputs.files]
    if missing:
        raise ValueError(
            f"Missing keys in NPZ file: {missing}. Found keys: {list(outputs.files)}"
        )


def _axis_conversion_matrix() -> np.ndarray:
    # Same convention as convert_egobody_to_egoallo.py:
    # HoloLens y-up -> EgoAllo z-up, x'=x, y'=-z, z'=y.
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )


def _make_homogeneous_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rotation.astype(np.float32)
    T[:3, 3] = translation.astype(np.float32)
    return T


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    flat = points.reshape(-1, 3)
    flat_out = flat @ T[:3, :3].T + T[:3, 3]
    return flat_out.reshape(points.shape).astype(np.float32)


def _load_kinect12_to_holo(egobody_root: Path, recording: str) -> np.ndarray:
    calib_path = (
        egobody_root
        / "calibrations"
        / recording
        / "cal_trans"
        / "holo_to_kinect12.json"
    )
    with calib_path.open("r") as f:
        holo_to_kinect = np.asarray(json.load(f)["trans"], dtype=np.float32)
    return np.linalg.inv(holo_to_kinect).astype(np.float32)


def _gt_points_transform(
    egobody_root: Path,
    recording: str,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return np.eye(4, dtype=np.float32)
    if mode == "kinect12_to_holo":
        return _load_kinect12_to_holo(egobody_root, recording)
    if mode == "holo_y_up_to_z_up":
        return _make_homogeneous_transform(_axis_conversion_matrix(), np.zeros(3))
    if mode == "kinect12_to_holo_zup":
        B4 = _make_homogeneous_transform(_axis_conversion_matrix(), np.zeros(3))
        return B4 @ _load_kinect12_to_holo(egobody_root, recording)
    raise ValueError(f"Unknown GT coordinate mode: {mode}")


def _recording_from_input(
    input_data: np.lib.npyio.NpzFile | None,
    npz_path: Path,
) -> str | None:
    if input_data is not None and "recording" in input_data.files:
        return _np_scalar_to_str(input_data["recording"])
    # Expected: .../egoallo_egobody_trajectories/<recording>/egoallo_outputs/foo.npz
    parent = npz_path.resolve().parent.parent
    if parent.name:
        return parent.name
    return None


def _parse_frame_id(path_like: object) -> int | None:
    stem = Path(str(path_like)).stem
    match = re.search(r"_frame_(\d+)$", stem)
    if match is None:
        match = re.search(r"frame_(\d+)$", stem)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_input_indices_for_outputs(
    outputs: np.lib.npyio.NpzFile,
    input_data: np.lib.npyio.NpzFile | None,
) -> np.ndarray:
    frame_nums = np.asarray(outputs["frame_nums"]).astype(np.int64)
    if input_data is None:
        return frame_nums
    if "timestamps_ns" in input_data.files and "timestamps_ns" in outputs.files:
        source_ts = np.asarray(input_data["timestamps_ns"], dtype=np.int64)
        target_ts = np.asarray(outputs["timestamps_ns"], dtype=np.int64)
        order = np.argsort(source_ts)
        sorted_ts = source_ts[order]
        right = np.searchsorted(sorted_ts, target_ts, side="left")
        left = np.clip(right - 1, 0, len(sorted_ts) - 1)
        right = np.clip(right, 0, len(sorted_ts) - 1)
        use_right = np.abs(sorted_ts[right] - target_ts) < np.abs(sorted_ts[left] - target_ts)
        return order[np.where(use_right, right, left)].astype(np.int64)
    return frame_nums


def _resolve_gt_frame_ids(
    outputs: np.lib.npyio.NpzFile,
    input_data: np.lib.npyio.NpzFile | None,
) -> np.ndarray:
    input_indices = _resolve_input_indices_for_outputs(outputs, input_data)
    if input_data is not None and "image_paths" in input_data.files:
        image_paths = np.asarray(input_data["image_paths"], dtype=object)
        if input_indices.size > 0 and np.max(input_indices) < len(image_paths):
            parsed = [_parse_frame_id(image_paths[int(i)]) for i in input_indices]
            if all(x is not None for x in parsed):
                return np.asarray(parsed, dtype=np.int64)
    return np.asarray(outputs["frame_nums"]).astype(np.int64)


def _find_gt_recording_root(
    egobody_root: Path,
    recording: str,
    split: str,
    role: str = "camera_wearer",
) -> Path | None:
    role_prefixes = {
        "camera_wearer": "smplx_camera_wearer",
        "interactee": "smplx_interactee",
    }
    if role not in role_prefixes:
        raise ValueError(f"Unknown EgoBody GT role: {role}")
    splits = [split] if split != "auto" else ["train", "val", "test"]
    for split_name in splits:
        candidate = egobody_root / f"{role_prefixes[role]}_{split_name}" / recording
        if candidate.exists():
            return candidate
    return None


def _find_gt_frame_map(gt_root: Path, body_idx: int | None) -> dict[int, Path]:
    if body_idx is None:
        body_dirs = sorted(gt_root.glob("body_idx_*"))
    else:
        body_dirs = [gt_root / f"body_idx_{body_idx}"]
    for body_dir in body_dirs:
        if not body_dir.exists():
            continue
        frame_map: dict[int, Path] = {}
        for p in body_dir.glob("results/frame_*/000.pkl"):
            frame_id = _parse_frame_id(p.parent.name)
            if frame_id is not None:
                frame_map[frame_id] = p
        if frame_map:
            print(f"Loaded EgoBody GT index: {body_dir} ({len(frame_map)} frames)")
            return frame_map
    return {}


def _load_gt_pkl(path: Path) -> dict[str, np.ndarray | str]:
    with path.open("rb") as f:
        return pickle.load(f)


def _stack_gt_param(
    frames: list[dict[str, np.ndarray | str]],
    key: str,
    width: int,
    device: torch.device,
    default: float = 0.0,
) -> torch.Tensor:
    values = []
    for frame in frames:
        if key in frame:
            arr = np.asarray(frame[key], dtype=np.float32).reshape(-1, width)[0]
        else:
            arr = np.full((width,), default, dtype=np.float32)
        values.append(arr)
    return torch.from_numpy(np.stack(values, axis=0)).to(device=device)


def _load_gt_mesh_sequence(
    pkl_paths: list[Path],
    model_path: Path,
    device: torch.device,
    zero_betas: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import smplx

    frames = [_load_gt_pkl(p) for p in pkl_paths]
    batch_size = len(frames)
    model = smplx.create(
        str(model_path),
        model_type="smplx",
        gender="neutral",
        use_pca=True,
        num_pca_comps=12,
        batch_size=batch_size,
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
            return_verts=True,
        )
    vertices = out.vertices.detach().cpu().numpy().astype(np.float32)
    joints = out.joints.detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int32)
    return vertices, joints, faces


def _normalize_vectors(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    out = np.broadcast_to(fallback.astype(np.float32), vec.shape).copy()
    valid = norm[..., 0] > 1e-8
    out[valid] = vec[valid] / norm[valid]
    return out


def _rotation_matrices_to_wxyz(rotmats: np.ndarray) -> np.ndarray:
    rotmats = np.asarray(rotmats, dtype=np.float32)
    quats = np.zeros(rotmats.shape[:-2] + (4,), dtype=np.float32)
    flat_r = rotmats.reshape(-1, 3, 3)
    flat_q = quats.reshape(-1, 4)
    for i, R in enumerate(flat_r):
        trace = float(np.trace(R))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            flat_q[i, 0] = 0.25 * s
            flat_q[i, 1] = (R[2, 1] - R[1, 2]) / s
            flat_q[i, 2] = (R[0, 2] - R[2, 0]) / s
            flat_q[i, 3] = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            flat_q[i, 0] = (R[2, 1] - R[1, 2]) / s
            flat_q[i, 1] = 0.25 * s
            flat_q[i, 2] = (R[0, 1] + R[1, 0]) / s
            flat_q[i, 3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            flat_q[i, 0] = (R[0, 2] - R[2, 0]) / s
            flat_q[i, 1] = (R[0, 1] + R[1, 0]) / s
            flat_q[i, 2] = 0.25 * s
            flat_q[i, 3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            flat_q[i, 0] = (R[1, 0] - R[0, 1]) / s
            flat_q[i, 1] = (R[0, 2] + R[2, 0]) / s
            flat_q[i, 2] = (R[1, 2] + R[2, 1]) / s
            flat_q[i, 3] = 0.25 * s
    quats /= np.maximum(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-8)
    return quats


def _estimate_gt_body_frames(joints_by_t: list[np.ndarray | None]) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    body_rots: list[np.ndarray | None] = []
    head_rots: list[np.ndarray | None] = []
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    x_fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    y_fallback = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    for joints in joints_by_t:
        if joints is None or joints.shape[0] <= 21:
            body_rots.append(None)
            head_rots.append(None)
            continue

        # Approximate SMPL-X body frame: +X is body right, +Z is world up,
        # +Y is body forward, matching EgoAllo/AMASS canonical comments.
        body_right = ((joints[2] - joints[1]) + (joints[17] - joints[16])) * 0.5
        body_right[2] = 0.0
        body_x = _normalize_vectors(body_right[None], x_fallback)[0]
        body_y = _normalize_vectors(np.cross(z_axis, body_x)[None], y_fallback)[0]
        body_x = _normalize_vectors(np.cross(body_y, z_axis)[None], x_fallback)[0]
        body_rots.append(np.stack([body_x, body_y, z_axis], axis=1).astype(np.float32))

        # Approximate head frame at the GT head joint: use shoulder midpoint to
        # head as local +Z, and the same body right as local +X.
        shoulder_mid = (joints[16] + joints[17]) * 0.5
        head_z = _normalize_vectors((joints[15] - shoulder_mid)[None], z_axis)[0]
        head_x = body_x - np.dot(body_x, head_z) * head_z
        head_x = _normalize_vectors(head_x[None], x_fallback)[0]
        head_y = _normalize_vectors(np.cross(head_z, head_x)[None], y_fallback)[0]
        head_x = _normalize_vectors(np.cross(head_y, head_z)[None], x_fallback)[0]
        head_rots.append(np.stack([head_x, head_y, head_z], axis=1).astype(np.float32))
    return body_rots, head_rots


def _csv_hand_path_points(
    outputs: np.lib.npyio.NpzFile,
    input_data: np.lib.npyio.NpzFile | None,
    side: str,
    point_name: str,
) -> np.ndarray | None:
    if input_data is None:
        return None
    key = f"{side}_hand_{point_name}_position"
    available_key = f"{side}_hand_available"
    if key not in input_data.files or available_key not in input_data.files:
        return None
    indices = _resolve_input_indices_for_outputs(outputs, input_data)
    if indices.size == 0 or np.max(indices) >= len(input_data[key]):
        return None
    points = np.asarray(input_data[key][indices], dtype=np.float32)
    available = np.asarray(input_data[available_key][indices], dtype=bool)
    finite = np.isfinite(points).all(axis=1)
    points = points[available & finite]
    if len(points) < 2:
        return None
    return points


def _make_csv_hand_side_detection(
    outputs: np.lib.npyio.NpzFile,
    input_data: np.lib.npyio.NpzFile | None,
    side: str,
    device: torch.device,
):
    if input_data is None:
        return None
    from egoallo.hand_detection_structs import AriaHandWristPoseWrtWorld

    required = [
        f"{side}_hand_available",
        f"{side}_hand_wrist_position",
        f"{side}_hand_palm_position",
        f"{side}_hand_palm_normal",
    ]
    if any(key not in input_data.files for key in required):
        return None
    input_indices = _resolve_input_indices_for_outputs(outputs, input_data)
    if input_indices.size == 0 or np.max(input_indices) >= len(input_data[required[0]]):
        return None

    available = np.asarray(input_data[f"{side}_hand_available"][input_indices], dtype=bool)
    wrists = np.asarray(input_data[f"{side}_hand_wrist_position"][input_indices], dtype=np.float32)
    palms = np.asarray(input_data[f"{side}_hand_palm_position"][input_indices], dtype=np.float32)
    normals = np.asarray(input_data[f"{side}_hand_palm_normal"][input_indices], dtype=np.float32)
    finite = np.isfinite(wrists).all(axis=1) & np.isfinite(palms).all(axis=1) & np.isfinite(normals).all(axis=1)
    valid = available & finite
    indices = np.flatnonzero(valid).astype(np.int64)
    if len(indices) == 0:
        return None
    normals = normals[indices]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return AriaHandWristPoseWrtWorld(
        confidence=torch.ones((len(indices),), dtype=torch.float32, device=device),
        wrist_position=torch.from_numpy(wrists[indices]).to(device),
        wrist_normal=torch.from_numpy(normals.astype(np.float32)).to(device),
        palm_position=torch.from_numpy(palms[indices]).to(device),
        palm_normal=torch.from_numpy(normals.astype(np.float32)).to(device),
        indices=torch.from_numpy(indices).to(device=device, dtype=torch.int64),
    )


def _make_csv_hand_detections_for_vis(
    outputs: np.lib.npyio.NpzFile,
    input_data: np.lib.npyio.NpzFile | None,
    device: torch.device,
):
    if input_data is None:
        return None
    from egoallo.hand_detection_structs import CorrespondedAriaHandWristPoseDetections

    left = _make_csv_hand_side_detection(outputs, input_data, "left", device)
    right = _make_csv_hand_side_detection(outputs, input_data, "right", device)
    if left is None and right is None:
        return None
    return CorrespondedAriaHandWristPoseDetections(
        detections_left_concat=left,
        detections_right_concat=right,
    )


class _GtOverlay:
    def __init__(
        self,
        server: viser.ViserServer,
        vertices_by_t: list[np.ndarray | None],
        joints_by_t: list[np.ndarray | None],
        faces: np.ndarray,
        input_cpf_points: np.ndarray,
        show_paths_initial: bool,
        mesh_mode: str,
        csv_left_wrist_points: np.ndarray | None = None,
        csv_right_wrist_points: np.ndarray | None = None,
        show_axes_initial: bool = True,
        label: str = "GT",
        scene_prefix: str = "/egobody_gt",
        debug_prefix: str = "/debug_paths/gt",
        color: tuple[int, int, int] = (52, 211, 153),
    ) -> None:
        self._server = server
        self._vertices_by_t = vertices_by_t
        self._joints_by_t = joints_by_t
        self._faces = faces
        self._mesh_mode = mesh_mode
        self._label = label
        self._scene_prefix = scene_prefix.rstrip("/")
        self._debug_prefix = debug_prefix.rstrip("/")
        self._color = color
        self._handles: list[viser.MeshHandle | None] = [None] * len(vertices_by_t)
        self._mesh_handle: viser.MeshHandle | None = None
        self._current_t: int | None = None
        self._gt_body_rots_by_t, self._gt_head_rots_by_t = _estimate_gt_body_frames(joints_by_t)
        self._gt_body_frame = server.scene.add_frame(
            f"{self._scene_prefix}/body_frame",
            show_axes=True,
            axes_length=0.16,
            axes_radius=0.006,
            visible=show_axes_initial,
        )
        self._gt_head_frame = server.scene.add_frame(
            f"{self._scene_prefix}/head_frame",
            show_axes=True,
            axes_length=0.10,
            axes_radius=0.004,
            visible=show_axes_initial,
        )

        with server.gui.add_folder(f"EgoBody GT {self._label}"):
            self._show_mesh = server.gui.add_checkbox(f"Show {self._label} mesh", True)
            self._show_axes = server.gui.add_checkbox(f"Show {self._label} axes", show_axes_initial)
            self._show_paths = server.gui.add_checkbox(
                "Show debug paths", show_paths_initial
            )
            self._gt_opacity = server.gui.add_slider(
                f"{self._label} opacity", initial_value=0.62, min=0.0, max=1.0, step=0.01
            )

        gt_head_points = np.asarray(
            [j[15] for j in joints_by_t if j is not None and j.shape[0] > 15],
            dtype=np.float32,
        )
        gt_pelvis_points = np.asarray(
            [j[0] for j in joints_by_t if j is not None and j.shape[0] > 0],
            dtype=np.float32,
        )
        gt_left_wrist_points = np.asarray(
            [j[20] for j in joints_by_t if j is not None and j.shape[0] > 20],
            dtype=np.float32,
        )
        gt_right_wrist_points = np.asarray(
            [j[21] for j in joints_by_t if j is not None and j.shape[0] > 21],
            dtype=np.float32,
        )
        self._path_handles: list[viser.SceneNodeHandle] = []
        if len(input_cpf_points) >= 2:
            self._path_handles.append(
                server.scene.add_spline_catmull_rom(
                    f"{self._debug_prefix}/input_cpf",
                    input_cpf_points.astype(np.float32),
                    line_width=3.0,
                    color=(0, 190, 255),
                    visible=show_paths_initial,
                )
            )
        if len(gt_head_points) >= 2:
            self._path_handles.append(
                server.scene.add_spline_catmull_rom(
                    f"{self._debug_prefix}/head",
                    gt_head_points,
                    line_width=3.0,
                    color=self._color,
                    visible=show_paths_initial,
                )
            )
        if len(gt_pelvis_points) >= 2:
            self._path_handles.append(
                server.scene.add_spline_catmull_rom(
                    f"{self._debug_prefix}/pelvis",
                    gt_pelvis_points,
                    line_width=2.0,
                    color=(250, 204, 21),
                    visible=show_paths_initial,
                )
            )
        if len(gt_left_wrist_points) >= 2:
            self._path_handles.append(
                server.scene.add_spline_catmull_rom(
                    f"{self._debug_prefix}/left_wrist",
                    gt_left_wrist_points,
                    line_width=2.0,
                    color=(34, 197, 94),
                    visible=show_paths_initial,
                )
            )
        if len(gt_right_wrist_points) >= 2:
            self._path_handles.append(
                server.scene.add_spline_catmull_rom(
                    f"{self._debug_prefix}/right_wrist",
                    gt_right_wrist_points,
                    line_width=2.0,
                    color=(16, 185, 129),
                    visible=show_paths_initial,
                )
            )
        if csv_left_wrist_points is not None and len(csv_left_wrist_points) >= 2:
            self._path_handles.append(
                server.scene.add_spline_catmull_rom(
                    f"{self._debug_prefix}/csv_left_wrist",
                    csv_left_wrist_points.astype(np.float32),
                    line_width=3.0,
                    color=(248, 113, 113),
                    visible=show_paths_initial,
                )
            )
        if csv_right_wrist_points is not None and len(csv_right_wrist_points) >= 2:
            self._path_handles.append(
                server.scene.add_spline_catmull_rom(
                    f"{self._debug_prefix}/csv_right_wrist",
                    csv_right_wrist_points.astype(np.float32),
                    line_width=3.0,
                    color=(96, 165, 250),
                    visible=show_paths_initial,
                )
            )

        if self._mesh_mode == "preload":
            print(f"Preloading EgoBody GT {self._label} mesh handles...")
            for t, vertices in enumerate(self._vertices_by_t):
                if vertices is not None:
                    self._handles[t] = self._create_mesh(t, vertices, visible=False)
            print(f"EgoBody GT {self._label} mesh handles ready.")

        @self._show_paths.on_update
        def _(_) -> None:
            for handle in self._path_handles:
                handle.visible = self._show_paths.value

        @self._show_axes.on_update
        def _(_) -> None:
            self._gt_body_frame.visible = self._show_axes.value and self._current_t is not None
            self._gt_head_frame.visible = self._show_axes.value and self._current_t is not None

        @self._gt_opacity.on_update
        def _(_) -> None:
            for handle in self._handles:
                if handle is not None:
                    handle.opacity = self._gt_opacity.value
            if self._mesh_handle is not None:
                self._mesh_handle.opacity = self._gt_opacity.value

    def _create_mesh(
        self,
        t: int,
        vertices: np.ndarray,
        visible: bool,
    ) -> viser.MeshHandle:
        return self._server.scene.add_mesh_simple(
            f"{self._scene_prefix}/person/{t}",
            vertices=vertices,
            faces=self._faces,
            color=self._color,
            opacity=self._gt_opacity.value,
            visible=visible,
        )

    def _remove_dynamic_mesh(self) -> None:
        if self._mesh_handle is not None:
            self._mesh_handle.remove()
            self._mesh_handle = None

    def update(self, t: int | None) -> None:
        if t is None:
            return
        t = int(t)
        if not (0 <= t < len(self._vertices_by_t)):
            self._hide_current()
            return

        vertices = self._vertices_by_t[t]
        joints = self._joints_by_t[t]
        self._update_axes(t, joints)
        if vertices is None or not self._show_mesh.value:
            self._hide_current(hide_axes=False)
            self._current_t = t
            return

        if self._mesh_mode == "dynamic":
            if self._current_t == t and self._mesh_handle is not None:
                self._mesh_handle.visible = True
                self._mesh_handle.opacity = self._gt_opacity.value
                return
            # Add the new mesh before removing the old one; this avoids a blank
            # frame, but repeated add/remove can still stutter during playback.
            new_handle = self._server.scene.add_mesh_simple(
                f"{self._scene_prefix}/person/current",
                vertices=vertices,
                faces=self._faces,
                color=self._color,
                opacity=self._gt_opacity.value,
                visible=True,
            )
            self._remove_dynamic_mesh()
            self._mesh_handle = new_handle
            self._current_t = t
            return

        # lazy/preload mode: create once, then only toggle visibility. This is
        # much less flickery than removing and re-adding every frame.
        handle = self._handles[t]
        if handle is None:
            handle = self._create_mesh(t, vertices, visible=True)
            self._handles[t] = handle
        else:
            handle.visible = True
            handle.opacity = self._gt_opacity.value

        if self._current_t is not None and self._current_t != t:
            old = self._handles[self._current_t]
            if old is not None:
                old.visible = False
        self._current_t = t

    def _update_axes(self, t: int, joints: np.ndarray | None) -> None:
        show = bool(self._show_axes.value and joints is not None)
        self._gt_body_frame.visible = show
        self._gt_head_frame.visible = show
        if not show or joints is None:
            return

        body_rot = self._gt_body_rots_by_t[t]
        head_rot = self._gt_head_rots_by_t[t]
        if body_rot is not None and joints.shape[0] > 0:
            self._gt_body_frame.position = joints[0].astype(np.float32)
            self._gt_body_frame.wxyz = _rotation_matrices_to_wxyz(body_rot[None])[0]
        if head_rot is not None and joints.shape[0] > 15:
            self._gt_head_frame.position = joints[15].astype(np.float32)
            self._gt_head_frame.wxyz = _rotation_matrices_to_wxyz(head_rot[None])[0]

    def _hide_current(self, hide_axes: bool = True) -> None:
        if self._mesh_mode == "dynamic":
            self._remove_dynamic_mesh()
        elif self._current_t is not None and 0 <= self._current_t < len(self._handles):
            handle = self._handles[self._current_t]
            if handle is not None:
                handle.visible = False
        if hide_axes:
            self._gt_body_frame.visible = False
            self._gt_head_frame.visible = False
        self._current_t = None

def _add_egobody_gt_overlay(
    server: viser.ViserServer,
    npz_path: Path,
    outputs: np.lib.npyio.NpzFile,
    input_data: np.lib.npyio.NpzFile | None,
    device: torch.device,
    *,
    egobody_root: Path,
    split: str,
    body_idx: int | None,
    smplx_model_path: Path,
    coordinate_mode: str,
    zero_betas: bool,
    show_paths: bool,
    mesh_mode: str,
    role: str,
    label: str,
    scene_prefix: str,
    debug_prefix: str,
    color: tuple[int, int, int],
    include_csv_paths: bool = False,
) -> _GtOverlay | None:
    recording = _recording_from_input(input_data, npz_path)
    if recording is None:
        print(f"[warn] Could not infer EgoBody recording; {label} GT overlay disabled.")
        return None

    gt_root = _find_gt_recording_root(egobody_root, recording, split, role=role)
    if gt_root is None:
        print(f"[warn] No {label} SMPL-X GT found for {recording}; GT overlay disabled.")
        return None

    frame_map = _find_gt_frame_map(gt_root, body_idx)
    if not frame_map:
        print(f"[warn] No {label} SMPL-X frame pkls under {gt_root}; GT overlay disabled.")
        return None

    gt_frame_ids = _resolve_gt_frame_ids(outputs, input_data)
    pkl_paths: list[Path] = []
    valid_output_indices: list[int] = []
    for t, frame_id in enumerate(gt_frame_ids):
        p = frame_map.get(int(frame_id))
        if p is not None:
            valid_output_indices.append(t)
            pkl_paths.append(p)

    if not pkl_paths:
        print(
            "[warn] Found GT directory but no matching frame ids. "
            f"First requested ids: {gt_frame_ids[:8].tolist()}"
        )
        return None

    print(
        f"Loading EgoBody {label} GT meshes: {len(pkl_paths)}/{len(gt_frame_ids)} "
        f"matched frames, coordinate_mode={coordinate_mode}, zero_betas={zero_betas}"
    )
    vertices, joints, faces = _load_gt_mesh_sequence(
        pkl_paths,
        smplx_model_path,
        device=device,
        zero_betas=zero_betas,
    )
    T_gt_to_vis = _gt_points_transform(egobody_root, recording, coordinate_mode)
    vertices = _transform_points(vertices, T_gt_to_vis)
    joints = _transform_points(joints, T_gt_to_vis)

    timesteps = outputs["Ts_world_cpf"].shape[0]
    vertices_by_t: list[np.ndarray | None] = [None] * timesteps
    joints_by_t: list[np.ndarray | None] = [None] * timesteps
    for local_i, t in enumerate(valid_output_indices):
        if 0 <= t < timesteps:
            vertices_by_t[t] = vertices[local_i]
            joints_by_t[t] = joints[local_i]

    input_cpf_points = np.asarray(outputs["Ts_world_cpf"][..., 4:7], dtype=np.float32)
    csv_left_wrist_points = _csv_hand_path_points(outputs, input_data, "left", "wrist") if include_csv_paths else None
    csv_right_wrist_points = _csv_hand_path_points(outputs, input_data, "right", "wrist") if include_csv_paths else None
    print(f"EgoBody {label} GT overlay ready.")
    return _GtOverlay(
        server,
        vertices_by_t,
        joints_by_t,
        faces,
        input_cpf_points,
        show_paths,
        mesh_mode,
        csv_left_wrist_points=csv_left_wrist_points,
        csv_right_wrist_points=csv_right_wrist_points,
        label=label,
        scene_prefix=scene_prefix,
        debug_prefix=debug_prefix,
        color=color,
    )



def _load_gvhmr_exo_window_metadata(
    npz_path: Path,
) -> tuple[np.lib.npyio.NpzFile, np.lib.npyio.NpzFile, Path, int, int]:
    data = np.load(npz_path, allow_pickle=True)
    required = {"vertices_world", "faces"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"{npz_path} missing keys: {sorted(missing)}")

    input_path: Path | None = None
    if "input_npz" in data.files:
        raw_input = _np_scalar_to_str(data["input_npz"])
        if raw_input and raw_input != "None":
            input_path = Path(raw_input)
    if input_path is None or not input_path.exists():
        # Expected path layout: <traj_root>/<recording>/gvhmr_exo/foo_exo_world.npz
        candidate = npz_path.resolve().parent.parent / "egobody_egoallo_input.npz"
        if candidate.exists():
            input_path = candidate
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(
            f"Could not locate source egobody_egoallo_input.npz for {npz_path}. "
            "Expected key 'input_npz' in the exo NPZ or a sibling recording cache."
        )

    input_data = np.load(input_path, allow_pickle=True)
    pred_len = int(np.asarray(data["vertices_world"]).shape[0])
    start = int(data["start_index"]) if "start_index" in data.files else 0
    end = int(data["end_index"]) if "end_index" in data.files else start + pred_len
    if end - start != pred_len:
        print(
            f"[warn] Exo NPZ start/end length ({start}:{end}) does not match "
            f"vertices length ({pred_len}); using {start}:{start + pred_len}."
        )
        end = start + pred_len
    return data, input_data, input_path, start, end


def _resolve_gt_frame_ids_for_input_slice(
    input_data: np.lib.npyio.NpzFile,
    start: int,
    end: int,
) -> np.ndarray:
    if "image_paths" in input_data.files:
        image_paths = np.asarray(input_data["image_paths"], dtype=object)
        if 0 <= start <= end <= len(image_paths):
            parsed = [_parse_frame_id(p) for p in image_paths[start:end]]
            if parsed and all(x is not None for x in parsed):
                return np.asarray(parsed, dtype=np.int64)
    return np.arange(start, end, dtype=np.int64)


def _input_slice_path_points(
    input_data: np.lib.npyio.NpzFile,
    start: int,
    end: int,
) -> np.ndarray:
    if "Ts_world_pv_mats" in input_data.files:
        mats = np.asarray(input_data["Ts_world_pv_mats"][start:end], dtype=np.float32)
        if mats.ndim == 3 and mats.shape[-2:] == (4, 4):
            return mats[:, :3, 3]
    if "Ts_world_cpf_mats" in input_data.files:
        mats = np.asarray(input_data["Ts_world_cpf_mats"][start:end], dtype=np.float32)
        if mats.ndim == 3 and mats.shape[-2:] == (4, 4):
            return mats[:, :3, 3]
    return np.zeros((max(end - start, 0), 3), dtype=np.float32)


def _add_egobody_exo_gt_overlay_from_gvhmr(
    server: viser.ViserServer,
    gvhmr_exo_npz: Path,
    device: torch.device,
    *,
    egobody_root: Path,
    split: str,
    body_idx: int | None,
    smplx_model_path: Path,
    coordinate_mode: str,
    zero_betas: bool,
    show_paths: bool,
    mesh_mode: str,
    label: str = "interactee",
    scene_prefix: str = "/egobody_gt/interactee",
    debug_prefix: str = "/debug_paths/interactee_gt",
    color: tuple[int, int, int] = (251, 146, 60),
) -> _GtOverlay | None:
    exo_data, input_data, input_path, start, end = _load_gvhmr_exo_window_metadata(
        gvhmr_exo_npz
    )
    recording = _recording_from_input(input_data, input_path)
    if recording is None:
        print(f"[warn] Could not infer EgoBody recording from {input_path}; {label} GT disabled.")
        return None

    gt_root = _find_gt_recording_root(egobody_root, recording, split, role="interactee")
    if gt_root is None:
        print(f"[warn] No {label} SMPL-X GT found for {recording}; GT overlay disabled.")
        return None

    frame_map = _find_gt_frame_map(gt_root, body_idx)
    if not frame_map:
        print(f"[warn] No {label} SMPL-X frame pkls under {gt_root}; GT overlay disabled.")
        return None

    gt_frame_ids = _resolve_gt_frame_ids_for_input_slice(input_data, start, end)
    pkl_paths: list[Path] = []
    valid_indices: list[int] = []
    for t, frame_id in enumerate(gt_frame_ids):
        p = frame_map.get(int(frame_id))
        if p is not None:
            valid_indices.append(t)
            pkl_paths.append(p)

    if not pkl_paths:
        print(
            "[warn] Found interactee GT directory but no matching frame ids for "
            f"GVHMR exo window {start}:{end}. First requested ids: {gt_frame_ids[:8].tolist()}"
        )
        return None

    print(
        f"Loading EgoBody {label} GT from GVHMR exo window: "
        f"recording={recording}, input={input_path}, window={start}:{end}, "
        f"matched={len(pkl_paths)}/{len(gt_frame_ids)}, coordinate_mode={coordinate_mode}, "
        f"zero_betas={zero_betas}"
    )
    vertices, joints, faces = _load_gt_mesh_sequence(
        pkl_paths,
        smplx_model_path,
        device=device,
        zero_betas=zero_betas,
    )
    T_gt_to_vis = _gt_points_transform(egobody_root, recording, coordinate_mode)
    vertices = _transform_points(vertices, T_gt_to_vis)
    joints = _transform_points(joints, T_gt_to_vis)

    timesteps = int(np.asarray(exo_data["vertices_world"]).shape[0])
    vertices_by_t: list[np.ndarray | None] = [None] * timesteps
    joints_by_t: list[np.ndarray | None] = [None] * timesteps
    for local_i, t in enumerate(valid_indices):
        if 0 <= t < timesteps:
            vertices_by_t[t] = vertices[local_i]
            joints_by_t[t] = joints[local_i]

    input_path_points = _input_slice_path_points(input_data, start, end)
    print(f"EgoBody {label} GT overlay ready for GVHMR exo window {start}:{end}.")
    return _GtOverlay(
        server,
        vertices_by_t,
        joints_by_t,
        faces,
        input_path_points,
        show_paths,
        mesh_mode,
        label=label,
        scene_prefix=scene_prefix,
        debug_prefix=debug_prefix,
        color=color,
    )


def load_and_visualize_exo_only(
    server: viser.ViserServer,
    gvhmr_exo_npz: Path,
    device: torch.device,
    *,
    egobody_root: Path = Path("/public/home/wenxin/egobody"),
    gt_split: str = "auto",
    exo_gt_body_idx: int | None = None,
    gt_smplx_model_path: Path = Path(
        "/public/home/wenxin/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
    ),
    gt_coordinate_mode: str = "kinect12_to_holo_zup",
    gt_zero_betas: bool = False,
    show_exo_gt: bool = True,
    show_gt_paths: bool = True,
    gt_mesh_mode: str = "lazy",
    gvhmr_exo_mesh_mode: str = "lazy",
) -> Callable[[], int]:
    exo_data, _input_data, input_path, start, end = _load_gvhmr_exo_window_metadata(
        gvhmr_exo_npz
    )
    timesteps = int(np.asarray(exo_data["vertices_world"]).shape[0])
    print(
        f"Loaded GVHMR exo-only view: {gvhmr_exo_npz}\n"
        f"source_input={input_path}, window={start}:{end}, timesteps={timesteps}"
    )

    exo_overlay = _GvhmrExoOverlay(
        server,
        gvhmr_exo_npz,
        frame_offset=0,
        mesh_mode=gvhmr_exo_mesh_mode,
    )
    gt_overlay = None
    if show_exo_gt:
        gt_overlay = _add_egobody_exo_gt_overlay_from_gvhmr(
            server,
            gvhmr_exo_npz,
            device,
            egobody_root=egobody_root,
            split=gt_split,
            body_idx=exo_gt_body_idx,
            smplx_model_path=gt_smplx_model_path,
            coordinate_mode=gt_coordinate_mode,
            zero_betas=gt_zero_betas,
            show_paths=show_gt_paths,
            mesh_mode=gt_mesh_mode,
        )

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=max(timesteps - 1, 0), step=1, initial_value=0
        )
        playing = server.gui.add_checkbox("Playing", True)
        fps = server.gui.add_slider("FPS", min=1.0, max=60.0, step=1.0, initial_value=15.0)
        server.gui.add_markdown(f"`GVHMR exo window: {start}:{end}`")

    def _update(t: int) -> None:
        exo_overlay.update(t)
        if gt_overlay is not None:
            gt_overlay.update(t)

    @frame_slider.on_update
    def _(_) -> None:
        _update(int(frame_slider.value))

    _update(0)
    last_tick = time.time()

    def loop_cb() -> int:
        nonlocal last_tick
        now = time.time()
        if playing.value and timesteps > 1 and now - last_tick >= 1.0 / float(fps.value):
            frame_slider.value = (int(frame_slider.value) + 1) % timesteps
            last_tick = now
        time.sleep(0.01)
        return int(frame_slider.value)

    return loop_cb


class _GvhmrExoOverlay:
    def __init__(
        self,
        server: viser.ViserServer,
        npz_path: Path,
        frame_offset: int = 0,
        mesh_mode: str = "lazy",
    ) -> None:
        data = np.load(npz_path, allow_pickle=True)
        required = {"vertices_world", "faces"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{npz_path} missing keys: {sorted(missing)}")
        self._server = server
        self._npz_path = npz_path
        self._vertices = data["vertices_world"].astype(np.float32)
        self._faces = data["faces"].astype(np.int32)
        self._frame_offset = int(frame_offset)
        self._mesh_mode = mesh_mode
        self._handles: list[viser.MeshHandle | None] = [None] * len(self._vertices)
        self._mesh_handle: viser.MeshHandle | None = None
        self._current_t: int | None = None

        with server.gui.add_folder("GVHMR Exo"):
            self._show_mesh = server.gui.add_checkbox("Show exo mesh", True)
            self._opacity = server.gui.add_slider(
                "Exo opacity", initial_value=0.72, min=0.0, max=1.0, step=0.01
            )
            server.gui.add_markdown(f"`{npz_path.name}`")

        if self._mesh_mode == "preload":
            print("Preloading GVHMR exo mesh handles...")
            for t, vertices in enumerate(self._vertices):
                self._handles[t] = self._create_mesh(t, vertices, visible=False)
            print("GVHMR exo mesh handles ready.")

        @self._opacity.on_update
        def _(_) -> None:
            for handle in self._handles:
                if handle is not None:
                    handle.opacity = self._opacity.value
            if self._mesh_handle is not None:
                self._mesh_handle.opacity = self._opacity.value

    def _create_mesh(self, t: int, vertices: np.ndarray, visible: bool) -> viser.MeshHandle:
        return self._server.scene.add_mesh_simple(
            f"/gvhmr_exo/person/{t}",
            vertices=vertices,
            faces=self._faces,
            color=(96, 165, 250),
            opacity=self._opacity.value,
            visible=visible,
        )

    def _remove_dynamic_mesh(self) -> None:
        if self._mesh_handle is not None:
            self._mesh_handle.remove()
            self._mesh_handle = None

    def update(self, t: int | None) -> None:
        if t is None:
            return
        exo_t = int(t) + self._frame_offset
        if not (0 <= exo_t < len(self._vertices)) or not self._show_mesh.value:
            self._hide_current()
            return

        vertices = self._vertices[exo_t]
        if self._mesh_mode == "dynamic":
            if self._current_t == exo_t and self._mesh_handle is not None:
                self._mesh_handle.visible = True
                self._mesh_handle.opacity = self._opacity.value
                return
            new_handle = self._server.scene.add_mesh_simple(
                "/gvhmr_exo/person/current",
                vertices=vertices,
                faces=self._faces,
                color=(96, 165, 250),
                opacity=self._opacity.value,
                visible=True,
            )
            self._remove_dynamic_mesh()
            self._mesh_handle = new_handle
            self._current_t = exo_t
            return

        handle = self._handles[exo_t]
        if handle is None:
            handle = self._create_mesh(exo_t, vertices, visible=True)
            self._handles[exo_t] = handle
        else:
            handle.visible = True
            handle.opacity = self._opacity.value

        if self._current_t is not None and self._current_t != exo_t:
            old = self._handles[self._current_t]
            if old is not None:
                old.visible = False
        self._current_t = exo_t

    def _hide_current(self) -> None:
        if self._mesh_handle is not None:
            self._mesh_handle.visible = False
        if self._current_t is not None and 0 <= self._current_t < len(self._handles):
            handle = self._handles[self._current_t]
            if handle is not None:
                handle.visible = False
        self._current_t = None

def load_and_visualize(
    server: viser.ViserServer,
    npz_path: Path,
    body_model,
    device: torch.device,
    *,
    floor_z_override: float | None = None,
    show_joints: bool = False,
    show_scene_points: bool = False,
    max_scene_points: int = 50000,
    show_gt: bool = True,
    show_exo_gt: bool = True,
    egobody_root: Path = Path("/public/home/wenxin/egobody"),
    gt_split: str = "auto",
    gt_body_idx: int | None = None,
    exo_gt_body_idx: int | None = None,
    gt_smplx_model_path: Path = Path(
        "/public/home/wenxin/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
    ),
    gt_coordinate_mode: str = "kinect12_to_holo_zup",
    gt_zero_betas: bool = False,
    show_gt_paths: bool = True,
    gt_mesh_mode: str = "lazy",
    gvhmr_exo_npz: Path | None = None,
    gvhmr_exo_frame_offset: int = 0,
    gvhmr_exo_mesh_mode: str = "lazy",
) -> Callable[[], int]:
    from egoallo.vis_helpers import visualize_traj_and_hand_detections

    outputs = np.load(npz_path, allow_pickle=True)
    _check_output_keys(outputs)

    input_data, source_path = _load_input_metadata(npz_path, outputs)
    floor_z = _resolve_floor_z(npz_path, outputs, input_data, floor_z_override)
    points_data = (
        _load_scene_points(input_data, max_scene_points) if show_scene_points else None
    )

    traj = _build_traj(outputs, device)
    Ts_world_cpf = torch.from_numpy(outputs["Ts_world_cpf"]).to(
        device=device, dtype=torch.float32
    )
    csv_hand_detections = _make_csv_hand_detections_for_vis(outputs, input_data, device)

    print(f"Loaded output: {npz_path}")
    if source_path is not None:
        print(f"Loaded source input metadata: {source_path}")
    print(
        f"timesteps={Ts_world_cpf.shape[0]}, samples={outputs['body_quats'].shape[0]}, "
        f"floor_z={floor_z:.4f}, scene_points={0 if points_data is None else len(points_data)}"
    )

    gt_overlays: list[_GtOverlay] = []
    gvhmr_exo_overlay = None
    if gvhmr_exo_npz is not None:
        try:
            gvhmr_exo_overlay = _GvhmrExoOverlay(
                server,
                gvhmr_exo_npz,
                frame_offset=gvhmr_exo_frame_offset,
                mesh_mode=gvhmr_exo_mesh_mode,
            )
            print(f"Loaded GVHMR exo overlay: {gvhmr_exo_npz}")
        except Exception as exc:
            print(f"[warn] Failed to add GVHMR exo overlay: {type(exc).__name__}: {exc}")

    def _post_update(t: int) -> None:
        for gt_overlay in gt_overlays:
            gt_overlay.update(t)
        if gvhmr_exo_overlay is not None:
            gvhmr_exo_overlay.update(t)

    base_loop_cb = visualize_traj_and_hand_detections(
        server,
        Ts_world_cpf,
        traj,
        body_model,
        hamer_detections=None,
        aria_detections=csv_hand_detections,
        points_data=points_data,
        splat_path=None,
        floor_z=floor_z,
        show_joints=show_joints,
        get_ego_video=None,
        post_update_callbacks=(_post_update,),
    )

    if show_gt:
        try:
            gt_overlay = _add_egobody_gt_overlay(
                server,
                npz_path,
                outputs,
                input_data,
                device,
                egobody_root=egobody_root,
                split=gt_split,
                body_idx=gt_body_idx,
                smplx_model_path=gt_smplx_model_path,
                coordinate_mode=gt_coordinate_mode,
                zero_betas=gt_zero_betas,
                show_paths=show_gt_paths,
                mesh_mode=gt_mesh_mode,
                role="camera_wearer",
                label="wearer",
                scene_prefix="/egobody_gt/wearer",
                debug_prefix="/debug_paths/wearer_gt",
                color=(52, 211, 153),
                include_csv_paths=True,
            )
            if gt_overlay is not None:
                gt_overlays.append(gt_overlay)
        except Exception as exc:
            print(f"[warn] Failed to add EgoBody wearer GT overlay: {type(exc).__name__}: {exc}")

    if show_exo_gt:
        try:
            if gvhmr_exo_npz is not None:
                exo_gt_overlay = _add_egobody_exo_gt_overlay_from_gvhmr(
                    server,
                    gvhmr_exo_npz,
                    device,
                    egobody_root=egobody_root,
                    split=gt_split,
                    body_idx=exo_gt_body_idx,
                    smplx_model_path=gt_smplx_model_path,
                    coordinate_mode=gt_coordinate_mode,
                    zero_betas=gt_zero_betas,
                    show_paths=show_gt_paths,
                    mesh_mode=gt_mesh_mode,
                )
            else:
                exo_gt_overlay = _add_egobody_gt_overlay(
                    server,
                    npz_path,
                    outputs,
                    input_data,
                    device,
                    egobody_root=egobody_root,
                    split=gt_split,
                    body_idx=exo_gt_body_idx,
                    smplx_model_path=gt_smplx_model_path,
                    coordinate_mode=gt_coordinate_mode,
                    zero_betas=gt_zero_betas,
                    show_paths=show_gt_paths,
                    mesh_mode=gt_mesh_mode,
                    role="interactee",
                    label="interactee",
                    scene_prefix="/egobody_gt/interactee",
                    debug_prefix="/debug_paths/interactee_gt",
                    color=(251, 146, 60),
                    include_csv_paths=False,
                )
            if exo_gt_overlay is not None:
                gt_overlays.append(exo_gt_overlay)
        except Exception as exc:
            print(f"[warn] Failed to add EgoBody interactee GT overlay: {type(exc).__name__}: {exc}")

    def loop_cb() -> int:
        return base_loop_cb()

    return loop_cb


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _get_file_list(search_root_dir: Path) -> list[str]:
    files = sorted(search_root_dir.glob("**/egoallo_outputs/*.npz"))
    return ["None"] + [str(p.relative_to(search_root_dir)) for p in files]


def main() -> None:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root-dir",
        type=Path,
        default=repo_root / "egoallo_egobody_trajectories",
        help="Root scanned for **/egoallo_outputs/*.npz when --npz-path is not set.",
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=None,
        help="Directly visualize one EgoBody EgoAllo output NPZ.",
    )
    parser.add_argument("--egoallo-src", type=Path, default=repo_root / "src")
    parser.add_argument(
        "--smplh-npz-path",
        type=Path,
        default=repo_root / "data" / "smplh" / "neutral" / "model.npz",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--floor-z", type=float, default=None)
    parser.add_argument(
        "--show-joints", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--show-scene-points", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-scene-points", type=int, default=50000)
    parser.add_argument("--show-gt", action=argparse.BooleanOptionalAction, default=True, help="Show camera wearer SMPL-X GT")
    parser.add_argument("--show-exo-gt", action=argparse.BooleanOptionalAction, default=True, help="Show interactee SMPL-X GT")
    parser.add_argument("--egobody-root", type=Path, default=Path("/public/home/wenxin/egobody"))
    parser.add_argument("--gt-split", choices=("auto", "train", "val", "test"), default="auto")
    parser.add_argument("--gt-body-idx", type=int, default=None, help="Camera wearer GT body_idx override")
    parser.add_argument("--exo-gt-body-idx", type=int, default=None, help="Interactee GT body_idx override")
    parser.add_argument(
        "--gt-smplx-model-path",
        type=Path,
        default=Path("/public/home/wenxin/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"),
    )
    parser.add_argument(
        "--gt-coordinate-mode",
        choices=("kinect12_to_holo_zup", "kinect12_to_holo", "holo_y_up_to_z_up", "none"),
        default="kinect12_to_holo_zup",
    )
    parser.add_argument("--gt-zero-betas", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-gt-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt-mesh-mode", choices=("lazy", "preload", "dynamic"), default="lazy")
    parser.add_argument("--gvhmr-exo-npz", type=Path, default=None, help="Optional NPZ exported by export_gvhmr_exo_for_egoallo.py")
    parser.add_argument("--gvhmr-exo-frame-offset", type=int, default=0)
    parser.add_argument("--gvhmr-exo-mesh-mode", choices=("lazy", "preload", "dynamic"), default="lazy")
    args = parser.parse_args()

    _maybe_add_egoallo_src(args.egoallo_src)
    os.chdir(repo_root)

    from egoallo import fncsmpl

    device = _device_from_arg(args.device)
    print(f"Using device: {device}")
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(dark_mode=True)

    common_kwargs = dict(
        floor_z_override=args.floor_z,
        show_joints=args.show_joints,
        show_scene_points=args.show_scene_points,
        max_scene_points=args.max_scene_points,
        show_gt=args.show_gt,
        show_exo_gt=args.show_exo_gt,
        egobody_root=args.egobody_root,
        gt_split=args.gt_split,
        gt_body_idx=args.gt_body_idx,
        exo_gt_body_idx=args.exo_gt_body_idx,
        gt_smplx_model_path=args.gt_smplx_model_path,
        gt_coordinate_mode=args.gt_coordinate_mode,
        gt_zero_betas=args.gt_zero_betas,
        show_gt_paths=args.show_gt_paths,
        gt_mesh_mode=args.gt_mesh_mode,
        gvhmr_exo_npz=args.gvhmr_exo_npz,
        gvhmr_exo_frame_offset=args.gvhmr_exo_frame_offset,
        gvhmr_exo_mesh_mode=args.gvhmr_exo_mesh_mode,
    )

    if args.npz_path is not None:
        npz_path = args.npz_path.resolve()
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        loop_cb = load_and_visualize(
            server,
            npz_path,
            body_model,
            device,
            **common_kwargs,
        )
        while True:
            loop_cb()

    if args.gvhmr_exo_npz is not None:
        gvhmr_exo_npz = args.gvhmr_exo_npz.resolve()
        if not gvhmr_exo_npz.exists():
            raise FileNotFoundError(gvhmr_exo_npz)
        loop_cb = load_and_visualize_exo_only(
            server,
            gvhmr_exo_npz,
            device,
            egobody_root=args.egobody_root,
            gt_split=args.gt_split,
            exo_gt_body_idx=args.exo_gt_body_idx,
            gt_smplx_model_path=args.gt_smplx_model_path,
            gt_coordinate_mode=args.gt_coordinate_mode,
            gt_zero_betas=args.gt_zero_betas,
            show_exo_gt=args.show_exo_gt,
            show_gt_paths=args.show_gt_paths,
            gt_mesh_mode=args.gt_mesh_mode,
            gvhmr_exo_mesh_mode=args.gvhmr_exo_mesh_mode,
        )
        while True:
            loop_cb()

    search_root_dir = args.search_root_dir.resolve()
    if not search_root_dir.exists():
        raise FileNotFoundError(search_root_dir)

    def get_file_list() -> list[str]:
        return _get_file_list(search_root_dir)

    options = get_file_list()
    if len(options) == 1:
        print(f"No NPZ files found under {search_root_dir}/**/egoallo_outputs/*.npz")
    file_dropdown = server.gui.add_dropdown("File", options=options)
    refresh_file_list = server.gui.add_button("Refresh File List")

    @refresh_file_list.on_click
    def _(_) -> None:
        file_dropdown.options = get_file_list()

    trajectory_folder = server.gui.add_folder("Trajectory")
    current_file = "None"
    loop_cb: Callable[[], int | None] = lambda: None

    while True:
        loop_cb()
        if current_file == file_dropdown.value:
            continue

        current_file = file_dropdown.value
        server.scene.reset()
        if current_file == "None":
            loop_cb = lambda: None
            continue

        trajectory_folder.remove()
        trajectory_folder = server.gui.add_folder("Trajectory")
        with trajectory_folder:
            npz_path = (search_root_dir / current_file).resolve()
            loop_cb = load_and_visualize(
                server,
                npz_path,
                body_model,
                device,
                **common_kwargs,
            )
            args_path = npz_path.parent / f"{npz_path.stem}_args.yaml"
            if args_path.exists():
                with server.gui.add_folder("Args"):
                    server.gui.add_markdown("```\n" + args_path.read_text() + "\n```")
            source = _source_input_path(npz_path, np.load(npz_path, allow_pickle=True))
            if source is not None:
                server.gui.add_markdown(f"Source input: `{_relative_or_absolute(source, repo_root)}`")


if __name__ == "__main__":
    main()
