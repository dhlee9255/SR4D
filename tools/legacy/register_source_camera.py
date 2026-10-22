#!/usr/bin/env python
"""Register the REAL source camera (the one that shot the original
monocular video) into the SAME canonical_human_world coordinate system
4DAnyone's already-generated target cameras (cameras.json) live in.

Does NOT re-run 4DAnyone diffusion, does NOT train FreeTimeGS/4DGS, and
NEVER overwrites cameras.json or the generated videos -- everything lands
in a brand-new --output-dir. See docs/source_camera_registration.md style
report written alongside the outputs for the full method.

Usage:
    python tools/register_source_camera.py \\
        --fdanyone-dir <run>/fdanyone/<clip> \\
        --motion-dir <run>/gvhmr/results/<clip> \\
        --source-video <original input video path> \\
        --output-dir data/source_camera_registration/<clip> \\
        --gvhmr-root 4DAnyone/third_party/GVHMR \\
        --model-dir <models root> \\
        --camera-mode static \\
        --optimize-focal \\
        [--use-silhouette] [--optimize-principal-point] \\
        [--dry-run | --inspect]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source_camera.correspondence import build_correspondences  # noqa: E402
from source_camera.diagnostics import diagnostic_dynamic_camera  # noqa: E402
from source_camera.evaluation import (  # noqa: E402
    Thresholds,
    compute_metrics,
    nearest_ring_camera,
    target_rig_sanity_check,
)
from source_camera.geometry import (  # noqa: E402
    STABLE_JOINT_NAMES,
    check_world_and_camera_frame_conventions,
    coco_to_goliath_index_map,
    load_target_rig,
    reconstruct_skeleton_camera,
    recompute_body_geometry,
)
from source_camera.initialization import initialize_global_incam_alignment, initialize_multiframe_pnp  # noqa: E402
from source_camera.optimize import refine_static_camera  # noqa: E402
from source_camera.schema import (  # noqa: E402
    write_cameras_with_source_json,
    write_metrics_json,
    write_report_markdown,
    write_source_camera_json,
)
from source_camera.visualize import build_contact_sheet, encode_video, plot_reprojection_error, render_overlay_frames  # noqa: E402

from fdanyone.motion.result import MotionResult  # noqa: E402
from fdanyone.skeleton.keypoints import VISIBLE_KEYPOINT_IDS  # noqa: E402
from fdanyone.video import decode_canonical_clip, iter_rgb_video  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fdanyone-dir", required=True, help="e.g. runs/<run>/fdanyone/<clip> (has cameras.json)")
    parser.add_argument("--motion-dir", required=True, help="e.g. runs/<run>/gvhmr/results/<clip> (has motion.json)")
    parser.add_argument("--source-video", required=True, help="the original monocular input video")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gvhmr-root", default="4DAnyone/third_party/GVHMR")
    parser.add_argument("--model-dir", default=None, help="root containing 4danyone/smplx_to_goliath70.pt")
    parser.add_argument("--regressor-path", default=None)
    parser.add_argument("--foreground-model-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--camera-mode", choices=["static", "diagnostic_dynamic"], default="static")
    parser.add_argument("--optimize-focal", action="store_true", default=True)
    parser.add_argument("--no-optimize-focal", dest="optimize_focal", action="store_false")
    parser.add_argument("--optimize-principal-point", action="store_true", default=False)
    parser.add_argument("--use-silhouette", action="store_true", default=False)
    parser.add_argument("--sanity-check-camera-id", type=int, default=None, help="defaults to front_camera_ids[0]")
    parser.add_argument("--sanity-check-frames", type=int, nargs="+", default=None)
    parser.add_argument(
        "--source-start-time", type=float, default=0.0,
        help="must match whatever start_time the original 4DAnyone run used to decode this source video "
        "(see its own inference command/logs) -- otherwise the decoded frames will not line up with "
        "MotionResult's source_frame_indices/source_pts and K_fullimg/observed_keypoints_2d.",
    )
    parser.add_argument("--dry-run", "--inspect", dest="inspect_only", action="store_true", default=False)
    return parser.parse_args(argv)


def _inspect(motion: MotionResult, fdanyone_dir: Path, source_video: Path) -> None:
    print("=== --dry-run / --inspect: data shapes and coordinate systems only ===")
    print(f"MotionResult: {motion.num_frames} frames, image {motion.image_width}x{motion.image_height}, "
          f"motion_world={motion.motion_world!r}, gvhmr_revision={motion.gvhmr_revision}")
    print(f"K_fullimg shape={tuple(motion.K_fullimg.shape)}, "
          f"observed_keypoints_2d shape={tuple(motion.observed_keypoints_2d.shape)}")
    print(f"source_frame_indices[0:5]={motion.source_frame_indices[:5]}, "
          f"frame_timestamps_sec[0:5]={motion.frame_timestamps_sec[:5]}")
    rig = load_target_rig(fdanyone_dir / "cameras.json")
    print(f"cameras.json: {len(rig.cameras)} target cameras, front_camera_ids={rig.front_camera_ids}")
    print("NOTE: the published cameras.json does NOT persist motion_world_to_canonical_world "
          "(that only lives in the skeleton-conditioning stage's scratch dir) -- this tool always "
          "recomputes it via fdanyone.skeleton.pipeline._body_geometry.")
    print(f"estimated_ring_center (derived from camera positions, reference-only)={rig.estimated_ring_center}")
    problems = check_world_and_camera_frame_conventions(rig)
    print(f"World/camera-frame convention check: {'OK' if not problems else 'PROBLEMS: ' + '; '.join(problems)}")
    print(f"Source video: {source_video} (exists={source_video.exists()})")
    print("coco_to_goliath index map:", coco_to_goliath_index_map())
    print("STABLE_JOINT_NAMES:", STABLE_JOINT_NAMES)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fdanyone_dir = Path(args.fdanyone_dir).expanduser().resolve()
    motion_dir = Path(args.motion_dir).expanduser().resolve()
    source_video = Path(args.source_video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    gvhmr_root = Path(args.gvhmr_root).expanduser().resolve()

    motion = MotionResult.load(motion_dir)

    if args.inspect_only:
        _inspect(motion, fdanyone_dir, source_video)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    clip_name = fdanyone_dir.name

    # --- Section 8: target-rig sanity check must pass BEFORE we trust
    # anything about the coordinate system we're about to optimize in.
    rig = load_target_rig(fdanyone_dir / "cameras.json")
    convention_problems = check_world_and_camera_frame_conventions(rig)
    if convention_problems:
        print("FATAL: target-rig coordinate convention check failed:")
        for problem in convention_problems:
            print(f"  - {problem}")
        print("Refusing to run source-camera optimization against an unverified coordinate system.")
        return 2

    print(f"[1/9] Recomputing body geometry via fdanyone.skeleton.pipeline._body_geometry (device={args.device}) ...")
    body = recompute_body_geometry(
        motion, gvhmr_root, args.device, regressor_path=args.regressor_path, model_dir=args.model_dir
    )
    # motion_world_to_canonical_world is NOT persisted anywhere in the
    # published run (only the skeleton-conditioning stage's scratch dir has
    # it, which is not kept) -- see geometry.TargetRig's docstring. The
    # target-rig sanity check below (section 8) is this tool's real
    # end-to-end validation of the recomputed transform instead.
    transform_drift = float("nan")

    print("[2/9] Decoding the original source video (same canonical decode GVHMR used) ...")
    clip = decode_canonical_clip(str(source_video), num_frames=motion.num_frames, start_time=args.source_start_time)
    try:
        # Reuses fdanyone's own frame-identity contract check (exact
        # per-frame source_index/source_pts/timestamp/file-identity match),
        # not just a width/height guess -- catches a wrong --source-video
        # or --source-start-time far more rigorously than section 4's
        # "auto-detect full-frame vs cropped-frame" concern alone would.
        motion.validate_against_clip(clip)
    except Exception as error:  # noqa: BLE001 - surfaced as a fatal, not silently ignored
        print(
            f"FATAL: decoded clip does not match this MotionResult's frame identity: {error}\n"
            "Check --source-video and --source-start-time against the original 4DAnyone run's inputs; "
            "refusing to proceed with a silently mismatched pixel space."
        )
        return 2
    source_frames = np.stack([frame.rgb for frame in clip.frames], axis=0)

    print("[3/9] Building 3D-2D correspondences (stable joints, confidence-filtered, name-mapped) ...")
    K_fullimg_mean = motion.K_fullimg.numpy().mean(axis=0).astype(np.float64)
    observed = motion.observed_keypoints_2d.numpy().astype(np.float64)
    correspondences = build_correspondences(
        body.keypoints_world,
        observed,
        image_width=motion.image_width,
        image_height=motion.image_height,
    )
    print(f"  {correspondences.points_3d.shape[0]} correspondences across "
          f"{len(set(correspondences.frame_index.tolist()))} frames.")

    print("[4/9] Initialization: Option A (global-incam Kabsch) vs Option B (multi-frame PnP) ...")
    goliath_indices_by_name = coco_to_goliath_index_map()
    init_a = initialize_global_incam_alignment(
        body.keypoints_world,
        body.keypoints_incam,
        body.motion_world_to_canonical_world,
        goliath_indices_by_name,
        correspondences,
        K_fullimg_mean,
    )
    init_b = initialize_multiframe_pnp(
        correspondences, K_fullimg_mean, motion.image_width, motion.image_height, seed_w2c=init_a.w2c
    )
    print(f"  Option A ({init_a.method}): mean error {init_a.mean_reprojection_error_px:.2f}px")
    print(f"  Option B ({init_b.method}): mean error {init_b.mean_reprojection_error_px:.2f}px")
    chosen = init_a if init_a.mean_reprojection_error_px <= init_b.mean_reprojection_error_px else init_b
    print(f"  Chosen initialization: {chosen.method}")

    print("[5/9] Static-camera refinement (Stage A -> B -> optional C) ...")
    stages = refine_static_camera(
        chosen.w2c,
        K_fullimg_mean,
        correspondences,
        optimize_focal=args.optimize_focal,
        optimize_principal_point=args.optimize_principal_point,
        image_width=motion.image_width,
    )
    final_stage = list(stages.values())[-1]
    for name, stage in stages.items():
        print(f"  {name}: mean {stage.mean_reprojection_error_px:.2f}px, median {stage.median_reprojection_error_px:.2f}px, "
              f"p90 {stage.p90_reprojection_error_px:.2f}px")

    print("[6/9] Metrics + nearest-ring-camera reference + target-rig sanity check ...")
    silhouette_iou = None
    silhouette_overlay_path = None
    if args.use_silhouette:
        silhouette_iou = _compute_silhouette_iou(
            motion, source_frames, body, final_stage.w2c, final_stage.K, args, output_dir, gvhmr_root
        )
        if silhouette_iou is not None:
            print(f"  silhouette IoU (evaluation only, not used to further refine R/t/K): {silhouette_iou:.3f}")
        silhouette_overlay_path = str(output_dir / "optional_silhouette_overlay.mp4")

    metrics = compute_metrics(
        final_stage.w2c,
        final_stage.K,
        correspondences,
        motion.image_height,
        pnp_inlier_ratio=chosen.inlier_ratio,
        silhouette_iou=silhouette_iou,
        thresholds=Thresholds(),
    )
    nearest = nearest_ring_camera(final_stage.w2c, rig)
    sanity_camera_id = args.sanity_check_camera_id if args.sanity_check_camera_id is not None else rig.front_camera_ids[0]
    sanity_frames = args.sanity_check_frames or sorted({0, motion.num_frames // 2, motion.num_frames - 1})
    skeleton_video_path = fdanyone_dir / "skeletons" / f"{sanity_camera_id:02d}.mp4"
    skeleton_frames = {}
    if skeleton_video_path.is_file():
        for index, frame in enumerate(iter_rgb_video(skeleton_video_path)):
            if index in sanity_frames:
                skeleton_frames[index] = frame
    skeleton_camera = reconstruct_skeleton_camera(rig.camera(sanity_camera_id), clip.height, clip.width)
    if skeleton_frames:
        sample_frame = next(iter(skeleton_frames.values()))
        if sample_frame.shape[:2] != (skeleton_camera.image_height, skeleton_camera.image_width):
            print(
                f"  WARNING: reconstructed skeleton camera is {skeleton_camera.image_width}x{skeleton_camera.image_height} "
                f"but the actual skeleton video frame is {sample_frame.shape[1]}x{sample_frame.shape[0]} -- "
                "sanity check may be unreliable."
            )
    sanity = target_rig_sanity_check(
        body.keypoints_world, VISIBLE_KEYPOINT_IDS, skeleton_camera, skeleton_frames
    )
    print(f"  Target-rig sanity check (camera {sanity_camera_id}): hit ratio {sanity.hit_ratio:.2%} "
          f"-> {'PASS' if sanity.passed else 'FAIL'}")
    print(f"  Nearest generated camera (reference only): id={nearest.camera_id}, "
          f"{nearest.yaw_angular_difference_degrees:.1f} deg away")
    print(f"  Overall status: {metrics.status} -- {metrics.status_reason}")

    diagnostic = None
    if args.camera_mode == "diagnostic_dynamic":
        print("[7/9] Diagnostic dynamic-camera mode (NOT used as the shipped camera) ...")
        diagnostic = diagnostic_dynamic_camera(correspondences, final_stage.K, final_stage.w2c)
        print(f"  Diagnostic verdict: {diagnostic.verdict} -- {diagnostic.verdict_detail}")

    print("[8/9] Writing visual verification artifacts ...")
    overlay_dir = output_dir / "_overlay_frames"
    frame_paths = render_overlay_frames(source_frames, observed, correspondences, final_stage.w2c, final_stage.K, overlay_dir)
    overlay_video_path = output_dir / "source_registration_overlay.mp4"
    encode_video(frame_paths, overlay_video_path, fps=int(round(float(motion.fps))))
    contact_sheet_path = output_dir / "source_registration_contact_sheet.png"
    build_contact_sheet(frame_paths, metrics.per_frame_median_error_px, contact_sheet_path)
    error_plot_path = output_dir / "reprojection_error_plot.png"
    plot_reprojection_error(metrics.per_frame_median_error_px, metrics.per_joint_median_error_px, error_plot_path)

    print("[9/9] Writing source_camera.json / cameras_with_source.json / registration_metrics.json / report ...")
    status_reasons = [metrics.status_reason, f"sanity_check={'PASS' if sanity.passed else 'FAIL'}: {sanity.detail}"]
    final_status = metrics.status if sanity.passed else "FAIL"

    write_source_camera_json(
        out_path=output_dir / "source_camera.json",
        image_width=motion.image_width,
        image_height=motion.image_height,
        K=final_stage.K,
        w2c=final_stage.w2c,
        camera_mode="static",
        registration_image_space="full_uncropped_canonical_frame (matches motion.image_width/height, K_fullimg, observed_keypoints_2d)",
        crop_transform=None,
        initialization_method=chosen.method,
        metrics=metrics,
        status=final_status,
        status_reasons=status_reasons,
    )
    write_cameras_with_source_json(
        out_path=output_dir / "cameras_with_source.json",
        rig=rig,
        source_camera_record={
            "role": "real_source",
            "camera_model": "PINHOLE",
            "K": [[float(v) for v in row] for row in final_stage.K],
            "world_to_camera": [[float(v) for v in row] for row in final_stage.w2c],
            "camera_to_world": [[float(v) for v in row] for row in np.linalg.inv(final_stage.w2c)],
            "image_width": motion.image_width,
            "image_height": motion.image_height,
            "status": final_status,
        },
    )
    write_metrics_json(
        output_dir / "registration_metrics.json",
        metrics,
        extra={
            "init_a_error_px": init_a.mean_reprojection_error_px,
            "init_b_error_px": init_b.mean_reprojection_error_px,
            "chosen_init": chosen.method,
            "nearest_ring_camera_id": nearest.camera_id,
            "nearest_ring_yaw_diff_degrees": nearest.yaw_angular_difference_degrees,
            "sanity_check_hit_ratio": sanity.hit_ratio,
            "sanity_check_passed": sanity.passed,
            "diagnostic_verdict": None if diagnostic is None else diagnostic.verdict,
            "transform_drift_vs_cameras_json": transform_drift,
        },
    )

    files_added = [
        str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else str(p)
        for p in [
            output_dir / "source_camera.json",
            output_dir / "cameras_with_source.json",
            output_dir / "registration_metrics.json",
            output_dir / "registration_report.md",
            overlay_video_path,
            contact_sheet_path,
            error_plot_path,
        ]
    ]
    write_report_markdown(
        out_path=output_dir / "registration_report.md",
        clip_name=clip_name,
        files_added=files_added,
        command=" ".join(sys.argv),
        keypoint_3d_source="fdanyone.skeleton.pipeline._body_geometry (recomputed; MHR70/Goliath70 regressor on SMPL-X vertices) "
        "-- keypoints_world (canonical) and keypoints_incam (GVHMR per-frame incam), neither persisted by the "
        "original pipeline, so recomputed via the exact same code path.",
        keypoint_2d_source="MotionResult.observed_keypoints_2d (COCO-17, VitPose, full uncropped canonical frame) "
        "paired via fdanyone.K_fullimg.",
        canonical_transform_note="motion_world_to_canonical_world was recomputed via "
        "fdanyone.skeleton.pipeline._body_geometry (the exact code path build_skeleton_conditioning uses) and "
        "applied to every 3D keypoint before optimization; raw SMPL-X global keypoints were never used directly. "
        "This transform is NOT persisted anywhere in a completed run (only cameras.json's TRIMMED payload is "
        "kept, see geometry.TargetRig's docstring), so it cannot be diffed against a stored value -- the "
        "target-rig sanity check (section 10) is this tool's real end-to-end validation of it instead.",
        crop_transform_note="K_fullimg and observed_keypoints_2d share one uncropped pixel space "
        "(both sized motion.image_width x motion.image_height) -- verified by construction in "
        "fdanyone/motion/gvhmr.py, so no crop-transform is needed between them for this data.",
        init_a_method=init_a.method,
        init_a_error_px=init_a.mean_reprojection_error_px,
        init_b_method=init_b.method,
        init_b_error_px=init_b.mean_reprojection_error_px,
        chosen_init=chosen.method,
        stage_results=stages,
        final_K=final_stage.K,
        final_w2c=final_stage.w2c,
        metrics=metrics,
        status=final_status,
        nearest_ring=nearest,
        sanity_check=sanity,
        diagnostic_verdict=None if diagnostic is None else diagnostic.verdict,
        diagnostic_detail=None if diagnostic is None else diagnostic.verdict_detail,
        overlay_video_path=str(overlay_video_path),
        contact_sheet_path=str(contact_sheet_path),
        error_plot_path=str(error_plot_path),
        silhouette_overlay_path=silhouette_overlay_path,
        twenty_four_view_recommendation=_recommendation_text(final_status),
    )

    print(f"\nDone. status={final_status}. See {output_dir / 'registration_report.md'}")
    return 0


def _recommendation_text(status: str) -> str:
    if status == "PASS":
        return (
            "The existing 24-view generation and its cameras.json are unaffected and reusable as-is. "
            "The registered source camera can now be added as a 25th, real, non-generated training view "
            "for downstream reconstruction, since it is in the same canonical_human_world and passed the "
            "target-rig sanity check."
        )
    if status == "WARN":
        return (
            "The existing 24-view generation is untouched and remains reusable on its own. The source "
            "camera is directionally correct but should be inspected via the overlay video before being "
            "trusted as a training view -- treat it as provisional."
        )
    return (
        "The existing 24-view generation is untouched and remains reusable on its own for everything it "
        "was already used for. The source-camera registration itself FAILED and must not be added as a "
        "training view in this state; see the overlay video and per-frame/per-joint error breakdown above "
        "before attempting a fix."
    )


def _compute_silhouette_iou(motion, source_frames, body, w2c, K, args, output_dir, gvhmr_root) -> float | None:
    """Evaluation-only silhouette IoU at the keypoint-refined camera (not a
    further optimization pass in this implementation -- see report note)."""

    try:
        import cv2

        from fdanyone.foreground import predict_foreground_masks
        from fdanyone.skeleton.pipeline import _body_geometry
        from fdanyone.geometry.crop import Crop
    except ImportError:
        return None

    masks = predict_foreground_masks(
        tuple(source_frames), args.foreground_model_path or _default_foreground_model_path(args.model_dir), args.device
    )
    # Re-derive vertices_world the same way _body_geometry does internally
    # by recomputing geometry once more with vertex-level output retained.
    from fdanyone.assets import resolve_regressor

    regressor = resolve_regressor(args.regressor_path, model_dir=args.model_dir or "models")
    full_geometry = _body_geometry(motion, regressor, gvhmr_root, args.device)
    sample_frames = sorted({0, motion.num_frames // 2, motion.num_frames - 1})
    ious = []
    overlay_frames = []
    for frame_index in sample_frames:
        vertices = full_geometry.vertices_world[frame_index]
        homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
        camera_points = (w2c[:3] @ homogeneous.T).T
        projected = (K @ camera_points.T).T
        xy = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
        rendered_mask = np.zeros((motion.image_height, motion.image_width), dtype=np.uint8)
        in_bounds = (
            (camera_points[:, 2] > 0)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < motion.image_width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < motion.image_height)
        )
        points = xy[in_bounds].astype(np.int32)
        if points.shape[0] >= 3:
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(rendered_mask, hull, 1)
        real_mask = (masks[frame_index] >= 64).astype(np.uint8)
        intersection = int((rendered_mask & real_mask).sum())
        union = int((rendered_mask | real_mask).sum())
        if union > 0:
            ious.append(intersection / union)
        overlay = np.stack([real_mask * 255, rendered_mask * 255, np.zeros_like(real_mask)], axis=-1).astype(np.uint8)
        overlay_frames.append(overlay)
    if overlay_frames:
        overlay_dir = output_dir / "_silhouette_overlay_frames"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        paths = []
        for i, frame in enumerate(overlay_frames):
            path = overlay_dir / f"{i:03d}.png"
            Image.fromarray(frame).save(path)
            paths.append(path)
        encode_video(paths, output_dir / "optional_silhouette_overlay.mp4", fps=1)
    return float(np.mean(ious)) if ious else None


def _default_foreground_model_path(model_dir: str | None) -> str:
    from fdanyone.assets import resolve_foreground_model

    return str(resolve_foreground_model(model_dir or "models"))


if __name__ == "__main__":
    raise SystemExit(main())
