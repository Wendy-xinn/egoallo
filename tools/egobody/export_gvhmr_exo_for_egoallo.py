#!/usr/bin/env python3
"""Export GVHMR incam predictions into EgoAllo/EgoBody world coordinates.

The output NPZ is deliberately simple so egobody_visualize_outputs.py can load it
without importing GVHMR: vertices_world, joints_world, faces, and metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _add_gvhmr_to_path(root: Path) -> None:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _camera_to_pv_matrix(mode: str) -> np.ndarray:
    if mode == "opencv_to_holo":
        # OpenCV camera: X right, Y down, Z forward.
        # EgoBody PV/Holo local pose: X right, Y up, Z backward.
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    if mode == "identity":
        return np.eye(3, dtype=np.float32)
    raise ValueError(f"Unknown camera convention: {mode}")


def _transform_points(T_world_pv: np.ndarray, points_cam: np.ndarray, camera_convention: str) -> np.ndarray:
    C = _camera_to_pv_matrix(camera_convention)
    points_pv = np.einsum("ij,tvj->tvi", C, points_cam.astype(np.float32))
    return (
        np.einsum("tij,tvj->tvi", T_world_pv[:, :3, :3].astype(np.float32), points_pv)
        + T_world_pv[:, None, :3, 3].astype(np.float32)
    )


def _infer_interval(args: argparse.Namespace, result_len: int) -> tuple[int, int]:
    if args.frame_meta is not None:
        meta = json.loads(args.frame_meta.read_text())
        return int(meta["start_index"]), int(meta["end_index"])
    start = int(args.start_index)
    end = start + result_len if args.traj_length is None else start + int(args.traj_length)
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True, help="EgoBody input NPZ produced by convert_egobody_to_egoallo.py")
    parser.add_argument("--gvhmr-results", type=Path, required=True, help="GVHMR hmr4d_results.pt")
    parser.add_argument("--gvhmr-root", type=Path, default=Path("/public/home/wenxin/GVHMR"))
    parser.add_argument("--frame-meta", type=Path, default=None, help="JSON written by run_gvhmr_on_egobody.py")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--traj-length", type=int, default=None)
    parser.add_argument("--camera-convention", choices=["opencv_to_holo", "identity"], default="opencv_to_holo")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Resolve user-provided paths before changing cwd to the GVHMR repo.
    # This keeps relative paths relative to the shell directory where the user
    # launched this script, not relative to /public/home/wenxin/GVHMR.
    args.input_npz = args.input_npz.resolve()
    args.gvhmr_results = args.gvhmr_results.resolve()
    args.output_npz = args.output_npz.resolve()
    if args.frame_meta is not None:
        args.frame_meta = args.frame_meta.resolve()

    gvhmr_root = args.gvhmr_root.resolve()
    _add_gvhmr_to_path(gvhmr_root)
    os.chdir(gvhmr_root)

    from hmr4d.utils.smplx_utils import make_smplx
    from einops import einsum

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but CUDA is not available")

    input_data = np.load(args.input_npz, allow_pickle=True)
    if "Ts_world_pv_mats" not in input_data.files:
        raise KeyError(f"{args.input_npz} does not contain Ts_world_pv_mats")

    pred = torch.load(args.gvhmr_results, map_location="cpu")
    if "smpl_params_incam" not in pred:
        raise KeyError(f"{args.gvhmr_results} does not contain smpl_params_incam")
    result_len = int(next(iter(pred["smpl_params_incam"].values())).shape[0])
    start, end = _infer_interval(args, result_len)
    if end - start != result_len:
        raise ValueError(f"GVHMR length {result_len} does not match interval [{start}, {end})")
    T_world_pv = input_data["Ts_world_pv_mats"][start:end].astype(np.float32)
    if len(T_world_pv) != result_len:
        raise ValueError(f"Input PV pose length {len(T_world_pv)} does not match GVHMR length {result_len}")

    smplx = make_smplx("supermotion").to(device)
    smplx2smpl = torch.load(gvhmr_root / "hmr4d/utils/body_model/smplx2smpl_sparse.pt", map_location=device)
    faces = np.asarray(make_smplx("smpl").faces, dtype=np.int32)
    J_regressor = torch.load(gvhmr_root / "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", map_location=device)

    with torch.no_grad():
        smpl_params = _to_device(pred["smpl_params_incam"], device)
        smplx_out = smplx(**smpl_params)
        verts_cam = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
        joints_cam = einsum(J_regressor, verts_cam, "j v, t v c -> t j c")
    verts_cam_np = verts_cam.detach().cpu().numpy().astype(np.float32)
    joints_cam_np = joints_cam.detach().cpu().numpy().astype(np.float32)

    vertices_world = _transform_points(T_world_pv, verts_cam_np, args.camera_convention)
    joints_world = _transform_points(T_world_pv, joints_cam_np, args.camera_convention)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        vertices_world=vertices_world.astype(np.float32),
        joints_world=joints_world.astype(np.float32),
        faces=faces,
        input_npz=str(args.input_npz.resolve()),
        gvhmr_results=str(args.gvhmr_results.resolve()),
        start_index=np.asarray(start, dtype=np.int64),
        end_index=np.asarray(end, dtype=np.int64),
        camera_convention=args.camera_convention,
    )
    print(f"Saved {args.output_npz}")
    print(f"vertices_world={vertices_world.shape}, joints_world={joints_world.shape}, faces={faces.shape}")


if __name__ == "__main__":
    main()
