#!/usr/bin/env python3
"""Visualize EgoAllo outputs produced from EgoBody-converted trajectories.

This bypasses EgoAllo's original Aria/VRS discovery path. It directly loads the
NPZ saved by tools/egobody/egobody_inference.py and reuses EgoAllo's mesh
visualizer with no VRS video, MPS trajectory, HaMeR detections, or Aria hand
tracking requirements.

For EgoBody debugging, it can also overlay the camera wearer's SMPL-X GT mesh.
The GT overlay is intentionally diagnostic: it keeps SMPL-X and SMPL-H as
separate meshes instead of forcing an early SMPL-X -> SMPL-H conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
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


def _resolve_gt_frame_ids(
    outputs: np.lib.npyio.NpzFile,
    input_data: np.lib.npyio.NpzFile | None,
) -> np.ndarray:
    frame_nums = np.asarray(outputs["frame_nums"]).astype(np.int64)
    if input_data is not None and "image_paths" in input_data.files:
        image_paths = np.asarray(input_data["image_paths"], dtype=object)
        if frame_nums.size > 0 and np.max(frame_nums) < len(image_paths):
            parsed = [_parse_frame_id(image_paths[int(i)]) for i in frame_nums]
            if all(x is not None for x in parsed):
                return np.asarray(parsed, dtype=np.int64)
    return frame_nums


def _find_gt_recording_root(
    egobody_root: Path,
    recording: str,
    split: str,
) -> Path | None:
    splits = [split] if split != "auto" else ["train", "val", "test"]
    for split_name in splits:
        candidate = egobody_root / f"smplx_camera_wearer_{split_name}" / recording
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
    ) -> None:
        self._server = server
        self._vertices_by_t = vertices_by_t
        self._faces = faces
        self._mesh_mode = mesh_mode
        self._handles: list[viser.MeshHandle | None] = [None] * len(vertices_by_t)
        self._mesh_handle: viser.MeshHandle | None = None
        self._current_t: int | None = None

        with server.gui.add_folder("EgoBody GT"):
            self._show_mesh = server.gui.add_checkbox("Show GT mesh", True)
            self._show_paths = server.gui.add_checkbox(
                "Show debug paths", show_paths_initial
            )
            self._gt_opacity = server.gui.add_slider(
                "GT opacity", initial_value=0.62, min=0.0, max=1.0, step=0.01
            )

        gt_head_points = np.asarray(
            [j[15] for j in joints_by_t if j is not None and j.shape[0] > 15],
            dtype=np.float32,
        )
        gt_pelvis_points = np.asarray(
            [j[0] for j in joints_by_t if j is not None and j.shape[0] > 0],
            dtype=np.float32,
        )
        self._path_handles: list[viser.SceneNodeHandle] = []
        if len(input_cpf_points) >= 2:
            self._path_handles.append(
                server.scene.add_spline_catmull_rom(
                    "/debug_paths/input_cpf",
                    input_cpf_points.astype(np.float32),
                    line_width=3.0,
                    color=(0, 190, 255),
                    visible=show_paths_initial,
                )
            )
        if len(gt_head_points) >= 2:
            self._path_handles.append(
                server.scene.add_spline_catmull_rom(
                    "/debug_paths/gt_head",
                    gt_head_points,
                    line_width=3.0,
                    color=(52, 211, 153),
                    visible=show_paths_initial,
                )
            )
        if len(gt_pelvis_points) >= 2:
            self._path_handles.append(
                server.scene.add_spline_catmull_rom(
                    "/debug_paths/gt_pelvis",
                    gt_pelvis_points,
                    line_width=2.0,
                    color=(250, 204, 21),
                    visible=show_paths_initial,
                )
            )

        if self._mesh_mode == "preload":
            print("Preloading EgoBody GT mesh handles...")
            for t, vertices in enumerate(self._vertices_by_t):
                if vertices is not None:
                    self._handles[t] = self._create_mesh(t, vertices, visible=False)
            print("EgoBody GT mesh handles ready.")

        @self._show_paths.on_update
        def _(_) -> None:
            for handle in self._path_handles:
                handle.visible = self._show_paths.value

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
            f"/egobody_gt/person/{t}",
            vertices=vertices,
            faces=self._faces,
            color=(52, 211, 153),
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
        if vertices is None or not self._show_mesh.value:
            self._hide_current()
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
                "/egobody_gt/person/current",
                vertices=vertices,
                faces=self._faces,
                color=(52, 211, 153),
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

    def _hide_current(self) -> None:
        if self._mesh_mode == "dynamic":
            self._remove_dynamic_mesh()
        elif self._current_t is not None and 0 <= self._current_t < len(self._handles):
            handle = self._handles[self._current_t]
            if handle is not None:
                handle.visible = False
        self._current_t = None

def _add_egobody_gt_overlay(
    server: viser.ViserServer,
    npz_path: Path,
    outputs: np.lib.npyio.NpzFile,
    input_data: np.lib.npyio.NpzFile | None,
    device: torch.device,
    egobody_root: Path,
    split: str,
    body_idx: int | None,
    smplx_model_path: Path,
    coordinate_mode: str,
    zero_betas: bool,
    show_paths: bool,
    mesh_mode: str,
) -> _GtOverlay | None:
    recording = _recording_from_input(input_data, npz_path)
    if recording is None:
        print("[warn] Could not infer EgoBody recording; GT overlay disabled.")
        return None

    gt_root = _find_gt_recording_root(egobody_root, recording, split)
    if gt_root is None:
        print(f"[warn] No camera wearer SMPL-X GT found for {recording}; GT overlay disabled.")
        return None

    frame_map = _find_gt_frame_map(gt_root, body_idx)
    if not frame_map:
        print(f"[warn] No SMPL-X frame pkls under {gt_root}; GT overlay disabled.")
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
        f"Loading EgoBody camera wearer GT meshes: {len(pkl_paths)}/{len(gt_frame_ids)} "
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
    print("EgoBody GT overlay ready.")
    return _GtOverlay(server, vertices_by_t, joints_by_t, faces, input_cpf_points, show_paths, mesh_mode)


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
    egobody_root: Path = Path("/public/home/wenxin/egobody"),
    gt_split: str = "auto",
    gt_body_idx: int | None = None,
    gt_smplx_model_path: Path = Path(
        "/public/home/wenxin/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
    ),
    gt_coordinate_mode: str = "kinect12_to_holo_zup",
    gt_zero_betas: bool = False,
    show_gt_paths: bool = True,
    gt_mesh_mode: str = "lazy",
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

    print(f"Loaded output: {npz_path}")
    if source_path is not None:
        print(f"Loaded source input metadata: {source_path}")
    print(
        f"timesteps={Ts_world_cpf.shape[0]}, samples={outputs['body_quats'].shape[0]}, "
        f"floor_z={floor_z:.4f}, scene_points={0 if points_data is None else len(points_data)}"
    )

    gt_overlay = None

    def _post_update(t: int) -> None:
        if gt_overlay is not None:
            gt_overlay.update(t)

    base_loop_cb = visualize_traj_and_hand_detections(
        server,
        Ts_world_cpf,
        traj,
        body_model,
        hamer_detections=None,
        aria_detections=None,
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
            )
        except Exception as exc:
            print(f"[warn] Failed to add EgoBody GT overlay: {type(exc).__name__}: {exc}")

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
    parser.add_argument("--show-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--egobody-root", type=Path, default=Path("/public/home/wenxin/egobody"))
    parser.add_argument("--gt-split", choices=("auto", "train", "val", "test"), default="auto")
    parser.add_argument("--gt-body-idx", type=int, default=None)
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
        egobody_root=args.egobody_root,
        gt_split=args.gt_split,
        gt_body_idx=args.gt_body_idx,
        gt_smplx_model_path=args.gt_smplx_model_path,
        gt_coordinate_mode=args.gt_coordinate_mode,
        gt_zero_betas=args.gt_zero_betas,
        show_gt_paths=args.show_gt_paths,
        gt_mesh_mode=args.gt_mesh_mode,
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
