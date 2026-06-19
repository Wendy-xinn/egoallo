#!/usr/bin/env python3
"""Run the EgoBody -> EgoAllo ego + GVHMR exo pipeline.

This is an orchestration script. It deliberately calls the smaller scripts in
this folder instead of duplicating their logic:

1. convert_egobody_to_egoallo.py
   EgoBody images/head-hand-eye/PV calibration -> compact EgoAllo input NPZ.
2. egobody_inference.py
   EgoAllo sampling for the camera wearer.
3. run_gvhmr_on_egobody.py
   EgoBody PV frames -> GVHMR input video -> GVHMR hmr4d_results.pt.
4. export_gvhmr_exo_for_egoallo.py
   GVHMR incam prediction -> EgoBody/EgoAllo world exo NPZ.
5. diagnose_gvhmr_exo_reprojection.py, optional
   Reprojection and root diagnostics.

The output manifest is the intended handoff point for the next optimization
stage: it records the ego NPZ, exo NPZ, shared input trajectory, frame metadata,
and exact commands used to generate them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_ego_python() -> Path:
    candidates = [
        Path('/public/home/wenxin/miniconda3/envs/ego/bin/python'),
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _default_gvhmr_python() -> Path:
    candidates = [
        Path('/public/home/wenxin/miniconda3/envs/gvhmr/bin/python'),
        Path('/public/home/wenxin/miniconda3/envs/GVHMR/bin/python'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _cmd_to_str(cmd: list[str | os.PathLike[str]]) -> str:
    return ' '.join(shlex.quote(str(x)) for x in cmd)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def _run_step(
    manifest: dict[str, Any],
    manifest_path: Path,
    name: str,
    cmd: list[str | os.PathLike[str]],
    cwd: Path,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> None:
    step = {
        'name': name,
        'cmd': [str(x) for x in cmd],
        'cmd_str': _cmd_to_str(cmd),
        'cwd': str(cwd),
        'status': 'dry_run' if dry_run else 'running',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    manifest.setdefault('steps', []).append(step)
    _write_manifest(manifest_path, manifest)

    print(f'\n[{name}]')
    print(step['cmd_str'])
    if dry_run:
        return

    try:
        subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True, env=env)
    except Exception as exc:
        step['status'] = 'failed'
        step['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        step['error'] = repr(exc)
        _write_manifest(manifest_path, manifest)
        raise
    step['status'] = 'completed'
    step['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    _write_manifest(manifest_path, manifest)


def _latest_npz(directory: Path, before: set[Path] | None = None) -> Path | None:
    if not directory.exists():
        return None
    files = [p for p in directory.glob('*.npz') if not p.name.endswith('_args.npz')]
    if before is not None:
        created = [p for p in files if p not in before]
        if created:
            files = created
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _add_common_convert_args(cmd: list[str], args: argparse.Namespace, min_frames: int) -> None:
    cmd += [
        '--cpf-position-mode', args.cpf_position_mode,
        '--cpf-rotation-source', args.cpf_rotation_source,
        '--head-cpf-rotation', args.head_cpf_rotation,
        '--axis-conversion', args.axis_conversion,
        '--min-frames', str(min_frames),
    ]
    if args.valid_only:
        cmd.append('--valid-only')
    if args.max_frames is not None:
        cmd += ['--max-frames', str(args.max_frames)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True, help='EgoBody recording name, e.g. recording_20210907_S02_S01_01')
    parser.add_argument('--egobody-root', type=Path, default=Path('/public/home/wenxin/egobody'))
    parser.add_argument('--egoallo-root', type=Path, default=_repo_root())
    parser.add_argument('--gvhmr-root', type=Path, default=Path('/public/home/wenxin/GVHMR'))
    parser.add_argument('--ego-python', type=Path, default=_default_ego_python())
    parser.add_argument('--gvhmr-python', type=Path, default=_default_gvhmr_python())

    parser.add_argument('--output-root', type=Path, default=_repo_root() / 'egoallo_egobody_trajectories')
    parser.add_argument('--traj-root', type=Path, default=None, help='Defaults to <output-root>/<recording>')
    parser.add_argument('--input-name', default='egobody_egoallo_input.npz')
    parser.add_argument('--run-name', default="pipeline", help='Name used for manifest directory and default GVHMR video suffix')

    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--traj-length', type=int, default=128)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--hand-guidance', choices=['none', 'csv_wrist'], default='csv_wrist')
    parser.add_argument('--csv-hand-confidence', type=float, default=1.0)
    parser.add_argument('--flip-hand-normal', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--egoallo-checkpoint-dir', type=Path, default=_repo_root() / 'egoallo_checkpoint_april13' / 'checkpoints_3000000')
    parser.add_argument('--smplh-npz-path', type=Path, default=_repo_root() / 'data' / 'smplh' / 'neutral' / 'model.npz')

    parser.add_argument('--video-name', default=None, help='GVHMR video/output name. Defaults to <recording>_<start>_<end>_<run-name>')
    parser.add_argument('--gvhmr-output-root', type=Path, default=None, help='Defaults to <traj-root>/gvhmr_exo')
    parser.add_argument('--camera-convention', choices=['opencv_to_holo', 'identity'], default='opencv_to_holo')
    parser.add_argument('--static-cam', action='store_true')
    parser.add_argument('--use-dpvo', action='store_true')
    parser.add_argument('--f-mm', type=int, default=None, help='Only used when --no-real-k is set')
    parser.add_argument('--no-real-k', action='store_true', help='Use GVHMR estimated/f_mm intrinsics instead of EgoBody PV K')

    parser.add_argument('--run-convert', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--run-egoallo', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--run-gvhmr', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--run-export', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--run-diagnose', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--run-optimize', action=argparse.BooleanOptionalAction, default=False, help='Run NPZ postprocess optimization after exporting GVHMR exo world')
    parser.add_argument('--egoallo-output', type=Path, default=None, help='Reuse an existing EgoAllo output NPZ and skip locating a newly sampled one')

    parser.add_argument('--force-convert', action='store_true')
    parser.add_argument('--force-egoallo', action='store_true', help='Re-run EgoAllo even if the deterministic window alias already exists')
    parser.add_argument('--force-gvhmr', action='store_true', help='Pass --force to GVHMR demo through the runner')
    parser.add_argument('--force-export', action='store_true')
    parser.add_argument('--force-optimize', action='store_true')
    parser.add_argument('--copy-egoallo-alias', action=argparse.BooleanOptionalAction, default=True, help='Copy EgoAllo timestamped output to a deterministic pipeline_runs/<video_name>_egoallo.npz alias')
    parser.add_argument('--dry-run', action='store_true')

    parser.add_argument('--cpf-position-mode', choices=['gaze_median', 'head_origin', 'manual'], default='gaze_median')
    parser.add_argument('--cpf-rotation-source', choices=['head', 'pv'], default='head')
    parser.add_argument('--head-cpf-rotation', choices=['smpl', 'identity'], default='smpl')
    parser.add_argument('--axis-conversion', choices=['holo_y_up_to_z_up', 'none'], default='holo_y_up_to_z_up')
    parser.add_argument('--valid-only', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--diagnose-overlays', type=int, default=6)
    parser.add_argument('--opt-method', choices=['translation_smooth_ground', 'interaction_v1', 'copy'], default='interaction_v1')
    parser.add_argument('--opt-root-smooth-sigma', type=float, default=2.0)
    parser.add_argument('--opt-ground-weight', type=float, default=0.0)
    parser.add_argument('--opt-max-horizontal-correction-m', type=float, default=0.08)
    parser.add_argument('--opt-max-vertical-correction-m', type=float, default=0.06)
    parser.add_argument('--opt-interaction-device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--opt-interaction-iters', type=int, default=180)
    parser.add_argument('--opt-interaction-lr', type=float, default=0.05)
    parser.add_argument('--opt-interaction-optimize-z', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--opt-interaction-max-yaw-deg', type=float, default=8.0)
    parser.add_argument('--opt-keypoint-conf-thresh', type=float, default=0.25)
    parser.add_argument('--opt-interaction-reproj-weight', type=float, default=0.2)
    parser.add_argument('--opt-interaction-hand-weight', type=float, default=0.25)
    parser.add_argument('--opt-interaction-floor-weight', type=float, default=0.2)
    parser.add_argument('--opt-interaction-contact-weight', type=float, default=0.5)
    parser.add_argument('--opt-interaction-skate-weight', type=float, default=2.0)
    parser.add_argument('--opt-interaction-contact-height-m', type=float, default=0.18)
    parser.add_argument('--opt-interaction-contact-vel-m', type=float, default=0.025)
    parser.add_argument('--opt-interaction-egoexo-hand-weight', type=float, default=0.05)
    parser.add_argument('--opt-interaction-collision-weight', type=float, default=0.05)
    parser.add_argument('--opt-interaction-collision-margin-m', type=float, default=0.08)
    parser.add_argument('--opt-interaction-head-collision-weight', type=float, default=0.05)
    parser.add_argument('--opt-interaction-head-collision-margin-m', type=float, default=0.18)
    parser.add_argument('--opt-interaction-root-accel-weight', type=float, default=1.0)
    parser.add_argument('--opt-interaction-root-z-accel-weight', type=float, default=3.0)
    parser.add_argument('--opt-interaction-delta-vel-weight', type=float, default=1.0)
    parser.add_argument('--opt-interaction-anchor-weight', type=float, default=0.5)
    parser.add_argument('--opt-interaction-smooth-weight', type=float, default=8.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    egoallo_root = args.egoallo_root.resolve()
    gvhmr_root = args.gvhmr_root.resolve()
    traj_root = (args.traj_root or (args.output_root / args.recording)).resolve()
    input_npz = traj_root / args.input_name
    start = int(args.start_index)
    length = int(args.traj_length)
    end = start + length
    run_name = args.run_name or 'realK'
    video_name = args.video_name or f'{args.recording}_{start:06d}_{end:06d}_{run_name}'
    gvhmr_output_root = (args.gvhmr_output_root or (traj_root / 'gvhmr_exo')).resolve()
    gvhmr_results = gvhmr_output_root / video_name / 'hmr4d_results.pt'
    frame_meta = gvhmr_output_root / f'{video_name}_egobody_frames.json'
    exo_world_npz = gvhmr_output_root / f'{video_name}_exo_world.npz'
    optimized_exo_world_npz = gvhmr_output_root / f'{video_name}_exo_world_optimized.npz'
    egoallo_outputs_dir = traj_root / 'egoallo_outputs'

    pipeline_runs_dir = traj_root / 'pipeline_runs'
    manifest_path = pipeline_runs_dir / f'{video_name}_manifest.json'
    egoallo_alias_npz = pipeline_runs_dir / f'{video_name}_egoallo.npz'
    manifest: dict[str, Any] = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'recording': args.recording,
        'backend': 'egobody',
        'config': {
            'start_index': start,
            'traj_length': length,
            'end_index': end,
            'hand_guidance': args.hand_guidance,
            'camera_convention': args.camera_convention,
            'uses_real_egobody_K': not args.no_real_k,
        },
        'paths': {
            'traj_root': str(traj_root),
            'input_npz': str(input_npz),
            'egoallo_outputs_dir': str(egoallo_outputs_dir),
            'gvhmr_output_root': str(gvhmr_output_root),
            'gvhmr_results': str(gvhmr_results),
            'frame_meta': str(frame_meta),
            'exo_world_npz': str(exo_world_npz),
            'optimized_exo_world_npz': str(optimized_exo_world_npz),
            'egoallo_alias_npz': str(egoallo_alias_npz),
            'manifest': str(manifest_path),
        },
        'optimization_handoff': {
            'description': 'Use egoallo_output_npz + exo_world_npz + input_npz + frame_meta as the first optimization-stage inputs.',
            'egoallo_output_npz': None,
            'egoallo_alias_npz': str(egoallo_alias_npz),
            'exo_world_npz': str(exo_world_npz),
            'optimized_exo_world_npz': str(optimized_exo_world_npz),
            'input_npz': str(input_npz),
            'frame_meta': str(frame_meta),
        },
        'steps': [],
    }
    _write_manifest(manifest_path, manifest)

    min_frames = start + length + 1
    convert_script = egoallo_root / 'tools' / 'egobody' / 'convert_egobody_to_egoallo.py'
    egoallo_infer_script = egoallo_root / 'tools' / 'egobody' / 'egobody_inference.py'
    gvhmr_runner_script = egoallo_root / 'tools' / 'egobody' / 'run_gvhmr_on_egobody.py'
    export_script = egoallo_root / 'tools' / 'egobody' / 'export_gvhmr_exo_for_egoallo.py'
    diagnose_script = egoallo_root / 'tools' / 'egobody' / 'diagnose_gvhmr_exo_reprojection.py'
    optimize_script = egoallo_root / 'tools' / 'egobody' / 'optimize_ego_exo_npz.py'

    if args.run_convert:
        if input_npz.exists() and not args.force_convert:
            print(f'[convert] skip existing {input_npz} (use --force-convert to overwrite)')
            manifest['steps'].append({'name': 'convert', 'status': 'skipped_existing', 'output': str(input_npz)})
            _write_manifest(manifest_path, manifest)
        else:
            cmd = [
                str(args.ego_python),
                str(convert_script),
                '--recording', args.recording,
                '--egobody-root', str(args.egobody_root),
                '--output-root', str(args.output_root),
                '--output-name', args.input_name,
            ]
            _add_common_convert_args(cmd, args, min_frames)
            _run_step(manifest, manifest_path, 'convert', cmd, egoallo_root, args.dry_run)

    if not args.dry_run and not input_npz.exists():
        raise FileNotFoundError(f'Missing converted input NPZ: {input_npz}')

    egoallo_output = args.egoallo_output.resolve() if args.egoallo_output is not None else None
    if args.run_egoallo and egoallo_output is None and egoallo_alias_npz.exists() and not args.force_egoallo:
        egoallo_output = egoallo_alias_npz.resolve()
        print(f'[egoallo_inference] reuse deterministic alias {egoallo_output} (use --force-egoallo to resample)')
        manifest['steps'].append({'name': 'egoallo_inference', 'status': 'reused_alias', 'output': str(egoallo_output)})
        _write_manifest(manifest_path, manifest)
    if args.run_egoallo and egoallo_output is None:
        before = set(egoallo_outputs_dir.glob('*.npz')) if egoallo_outputs_dir.exists() else set()
        cmd = [
            str(args.ego_python),
            str(egoallo_infer_script),
            '--traj-root', str(traj_root),
            '--input-name', args.input_name,
            '--start-index', str(start),
            '--traj-length', str(length),
            '--num-samples', str(args.num_samples),
            '--hand-guidance', args.hand_guidance,
            '--csv-hand-confidence', str(args.csv_hand_confidence),
            '--checkpoint-dir', str(args.egoallo_checkpoint_dir),
            '--smplh-npz-path', str(args.smplh_npz_path),
        ]
        if args.flip_hand_normal:
            cmd.append('--flip-hand-normal')
        _run_step(manifest, manifest_path, 'egoallo_inference', cmd, egoallo_root, args.dry_run)
        if not args.dry_run:
            egoallo_output = _latest_npz(egoallo_outputs_dir, before)
            if egoallo_output is None:
                raise RuntimeError(f'EgoAllo inference completed but no new NPZ was found in {egoallo_outputs_dir}')
    elif egoallo_output is not None:
        print(f'[egoallo_inference] reuse {egoallo_output}')
        manifest['steps'].append({'name': 'egoallo_inference', 'status': 'reused_existing', 'output': str(egoallo_output)})
        _write_manifest(manifest_path, manifest)

    if egoallo_output is not None:
        egoallo_for_handoff = egoallo_output
        if args.copy_egoallo_alias:
            if args.dry_run:
                egoallo_for_handoff = egoallo_alias_npz
            else:
                egoallo_alias_npz.parent.mkdir(parents=True, exist_ok=True)
                if egoallo_output.resolve() != egoallo_alias_npz.resolve():
                    shutil.copy2(egoallo_output, egoallo_alias_npz)
                egoallo_for_handoff = egoallo_alias_npz
        manifest['paths']['egoallo_output_npz'] = str(egoallo_output)
        manifest['paths']['egoallo_alias_npz'] = str(egoallo_for_handoff)
        manifest['optimization_handoff']['egoallo_output_npz'] = str(egoallo_for_handoff)
        manifest['optimization_handoff']['egoallo_raw_output_npz'] = str(egoallo_output)
        _write_manifest(manifest_path, manifest)

    if args.run_gvhmr:
        if gvhmr_results.exists() and not args.force_gvhmr:
            print(f'[gvhmr] skip existing {gvhmr_results} (use --force-gvhmr to recompute)')
            manifest['steps'].append({'name': 'gvhmr', 'status': 'skipped_existing', 'output': str(gvhmr_results)})
            _write_manifest(manifest_path, manifest)
        else:
            cmd = [
                str(args.ego_python),
                str(gvhmr_runner_script),
                '--traj-root', str(traj_root),
                '--input-name', args.input_name,
                '--gvhmr-root', str(gvhmr_root),
                '--python', str(args.gvhmr_python),
                '--start-index', str(start),
                '--traj-length', str(length),
                '--video-name', video_name,
                '--output-root', str(gvhmr_output_root),
            ]
            if args.static_cam:
                cmd.append('--static-cam')
            if args.use_dpvo:
                cmd.append('--use-dpvo')
            if args.no_real_k:
                cmd.append('--no-real-k')
                if args.f_mm is not None:
                    cmd += ['--f-mm', str(args.f_mm)]
            if args.force_gvhmr:
                cmd.append('--force')
            _run_step(manifest, manifest_path, 'gvhmr', cmd, egoallo_root, args.dry_run)

    if not args.dry_run and args.run_export and not gvhmr_results.exists():
        raise FileNotFoundError(f'Missing GVHMR result: {gvhmr_results}')

    if args.run_export:
        if exo_world_npz.exists() and not args.force_export:
            print(f'[export_exo] skip existing {exo_world_npz} (use --force-export to overwrite)')
            manifest['steps'].append({'name': 'export_exo', 'status': 'skipped_existing', 'output': str(exo_world_npz)})
            _write_manifest(manifest_path, manifest)
        else:
            cmd = [
                str(args.gvhmr_python),
                str(export_script),
                '--input-npz', str(input_npz),
                '--gvhmr-results', str(gvhmr_results),
                '--gvhmr-root', str(gvhmr_root),
                '--frame-meta', str(frame_meta),
                '--camera-convention', args.camera_convention,
                '--output-npz', str(exo_world_npz),
            ]
            _run_step(manifest, manifest_path, 'export_exo', cmd, egoallo_root, args.dry_run)

    if args.run_optimize:
        if optimized_exo_world_npz.exists() and not args.force_optimize:
            print(f'[optimize_npz] skip existing {optimized_exo_world_npz} (use --force-optimize to overwrite)')
            manifest['steps'].append({'name': 'optimize_npz', 'status': 'skipped_existing', 'output': str(optimized_exo_world_npz)})
            _write_manifest(manifest_path, manifest)
        else:
            cmd = [
                str(args.ego_python),
                str(optimize_script),
                '--input-npz', str(input_npz),
                '--exo-world-npz', str(exo_world_npz),
                '--output-npz', str(optimized_exo_world_npz),
                '--method', args.opt_method,
                '--root-smooth-sigma', str(args.opt_root_smooth_sigma),
                '--ground-weight', str(args.opt_ground_weight),
                '--max-horizontal-correction-m', str(args.opt_max_horizontal_correction_m),
                '--max-vertical-correction-m', str(args.opt_max_vertical_correction_m),
            ]
            if args.opt_method == 'interaction_v1':
                cmd += [
                    '--egobody-root', str(args.egobody_root),
                    '--camera-convention', args.camera_convention,
                    '--smplh-npz-path', str(args.smplh_npz_path),
                    '--egoallo-src', str(args.egoallo_root / 'src'),
                    '--interaction-device', args.opt_interaction_device,
                    '--interaction-iters', str(args.opt_interaction_iters),
                    '--interaction-lr', str(args.opt_interaction_lr),
                    '--interaction-max-yaw-deg', str(args.opt_interaction_max_yaw_deg),
                    '--keypoint-conf-thresh', str(args.opt_keypoint_conf_thresh),
                    '--interaction-reproj-weight', str(args.opt_interaction_reproj_weight),
                    '--interaction-hand-weight', str(args.opt_interaction_hand_weight),
                    '--interaction-floor-weight', str(args.opt_interaction_floor_weight),
                    '--interaction-contact-weight', str(args.opt_interaction_contact_weight),
                    '--interaction-skate-weight', str(args.opt_interaction_skate_weight),
                    '--interaction-contact-height-m', str(args.opt_interaction_contact_height_m),
                    '--interaction-contact-vel-m', str(args.opt_interaction_contact_vel_m),
                    '--interaction-egoexo-hand-weight', str(args.opt_interaction_egoexo_hand_weight),
                    '--interaction-collision-weight', str(args.opt_interaction_collision_weight),
                    '--interaction-collision-margin-m', str(args.opt_interaction_collision_margin_m),
                    '--interaction-head-collision-weight', str(args.opt_interaction_head_collision_weight),
                    '--interaction-head-collision-margin-m', str(args.opt_interaction_head_collision_margin_m),
                    '--interaction-root-accel-weight', str(args.opt_interaction_root_accel_weight),
                    '--interaction-root-z-accel-weight', str(args.opt_interaction_root_z_accel_weight),
                    '--interaction-delta-vel-weight', str(args.opt_interaction_delta_vel_weight),
                    '--interaction-anchor-weight', str(args.opt_interaction_anchor_weight),
                    '--interaction-smooth-weight', str(args.opt_interaction_smooth_weight),
                ]
                if not args.opt_interaction_optimize_z:
                    cmd.append('--no-interaction-optimize-z')
            if egoallo_output is not None:
                cmd += ['--egoallo-npz', str(manifest['optimization_handoff'].get('egoallo_output_npz', egoallo_output))]
            _run_step(manifest, manifest_path, 'optimize_npz', cmd, egoallo_root, args.dry_run)
        manifest['optimization_handoff']['optimized_exo_world_npz'] = str(optimized_exo_world_npz)
        _write_manifest(manifest_path, manifest)

    if args.run_diagnose:
        diagnose_target = optimized_exo_world_npz if args.run_optimize else exo_world_npz
        cmd = [
            str(args.ego_python),
            str(diagnose_script),
            '--exo-world-npz', str(diagnose_target),
            '--num-overlays', str(args.diagnose_overlays),
        ]
        _run_step(manifest, manifest_path, 'diagnose_exo_reprojection', cmd, egoallo_root, args.dry_run)

    manifest['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    manifest['paths']['egoallo_output_npz'] = str(egoallo_output) if egoallo_output is not None else manifest['paths'].get('egoallo_output_npz')
    if egoallo_output is not None and not manifest['optimization_handoff'].get('egoallo_output_npz'):
        manifest['optimization_handoff']['egoallo_output_npz'] = str(egoallo_output)
    _write_manifest(manifest_path, manifest)

    print('\nPipeline manifest:')
    print(manifest_path)
    print('Optimization handoff:')
    print(json.dumps(manifest['optimization_handoff'], indent=2))


if __name__ == '__main__':
    main()
