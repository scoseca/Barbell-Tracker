from pathlib import Path

import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.barbell_tracker.analysis.squat_analyzer import SquatAnalyzer
from src.barbell_tracker.analysis.trajectory_analysis import TrajectoryAnalyzer
from src.barbell_tracker.export.export_to_csv import export_to_csv
from src.barbell_tracker.pose.foot_detector import FootPositionDetector
from src.barbell_tracker.tracking.barbell_tracker import BarbellTracker
from src.barbell_tracker.visualization.visualizer import Visualizer


def run_pipeline(config):
    """Run the full barbell tracking and squat analysis pipeline."""
    input_video = config["paths"]["input_video"]
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "started",
        "input_video": input_video,
        "tracked_objects": 0,
        "track_id": None,
        "trajectory_points": 0,
        "fps": None,
        "foot_position": None,
        "trajectory_analysis": None,
        "squat_analysis": None,
        "outputs": {},
        "warnings": [],
    }

    pose_landmarker = _create_pose_landmarker(config)
    tracker = BarbellTracker(
        model_path=config["models"]["yolo_model_path"],
        confidence=config.get("runtime", {}).get("confidence", 0.5),
        device=_select_device(config),
    )

    trajectories = tracker.process_video(input_video)
    if not trajectories:
        summary["status"] = "no_trajectories"
        summary["warnings"].append("No trajectories were detected.")
        return summary

    summary["tracked_objects"] = len(trajectories)
    track_id, trajectory = tracker.get_longest_trajectory()
    if track_id is None or not trajectory:
        summary["status"] = "no_trajectory_selected"
        summary["warnings"].append("No valid trajectory could be selected.")
        return summary

    summary["track_id"] = track_id
    summary["trajectory_points"] = len(trajectory)
    fps = _get_fps(config, tracker, input_video)
    summary["fps"] = fps

    analyzer = TrajectoryAnalyzer(fps=fps)
    trajectory_xy = [(point[0], point[1]) for point in trajectory]
    smoothing = config.get("analysis", {}).get("smoothing", {})
    smooth_trajectory = analyzer.smooth_trajectory(
        trajectory_xy,
        window_size=smoothing.get("window_size", 15),
        poly_order=smoothing.get("poly_order", 3),
    )
    analysis_results = analyzer.analyze_trajectory(trajectory_xy)
    if analysis_results is None:
        summary["status"] = "trajectory_too_short"
        summary["warnings"].append("Trajectory is too short for analysis.")
        return summary

    summary["trajectory_analysis"] = _summarize_trajectory_analysis(analysis_results)

    visualizer = Visualizer()
    foot_position = _detect_foot_position(config, input_video, pose_landmarker, summary)
    if foot_position is not None:
        summary["foot_position"] = tuple(float(value) for value in foot_position)
        squat_analysis = _analyze_squat(config, smooth_trajectory, foot_position)
        summary["squat_analysis"] = _summarize_squat_analysis(squat_analysis)
    else:
        squat_analysis = None
        summary["warnings"].append("Foot position could not be detected.")

    _write_outputs(
        config,
        visualizer,
        input_video,
        smooth_trajectory,
        analysis_results,
        foot_position,
        squat_analysis,
        summary,
    )

    summary["status"] = "completed"
    return summary


def _create_pose_landmarker(config):
    pose_config = config.get("pose", {})
    base_options = python.BaseOptions(
        model_asset_path=config["models"]["pose_landmarker_path"]
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=pose_config.get("output_segmentation_masks", False),
        min_pose_detection_confidence=pose_config.get("min_pose_detection_confidence", 0.5),
        min_tracking_confidence=pose_config.get("min_tracking_confidence", 0.5),
        min_pose_presence_confidence=pose_config.get("min_pose_presence_confidence", 0.5),
        num_poses=pose_config.get("num_poses", 1),
    )
    return vision.PoseLandmarker.create_from_options(options)


def _select_device(config):
    runtime = config.get("runtime", {})
    device = runtime.get("device", "cuda")
    if device == "cuda" and runtime.get("fallback_to_cpu", True):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except ImportError:
            return "cpu"
    return device


def _get_fps(config, tracker, input_video):
    configured_fps = config.get("video", {}).get("fps")
    if configured_fps:
        return configured_fps

    _, _, detected_fps = tracker.get_video_properties(input_video)
    return detected_fps or 30


def _detect_foot_position(config, input_video, pose_landmarker, summary):
    foot_config = config.get("foot_detection", {})
    detector = FootPositionDetector()
    try:
        return detector.extract_foot_positions_from_video(
            input_video,
            pose_landmarker,
            n_clusters=foot_config.get("n_clusters", 3),
            visualize=foot_config.get("visualize_clusters", True),
            plot_save_path=foot_config.get("clusters_plot_path"),
        )
    except Exception as exc:
        summary["warnings"].append(f"Foot detection failed: {exc}")
        return None


def _analyze_squat(config, smooth_trajectory, foot_position):
    squat_config = config.get("analysis", {}).get("squat", {})
    squat_analyzer = SquatAnalyzer(safe_zone_px=squat_config.get("safe_zone_px", 50))
    return squat_analyzer.analyze_squat(smooth_trajectory, foot_position)


def _write_outputs(
    config,
    visualizer,
    input_video,
    smooth_trajectory,
    analysis_results,
    foot_position,
    squat_analysis,
    summary,
):
    outputs = config.get("outputs", {})
    visualization = config.get("visualization", {})

    trajectory_plot_path = outputs.get(
        "trajectory_plot_path",
        str(Path(config.get("paths", {}).get("output_dir", "outputs")) / "barbell_position.png"),
    )
    _ensure_parent_dir(trajectory_plot_path)
    visualizer.plot_trajectory_2d(smooth_trajectory, save_path=trajectory_plot_path)
    summary["outputs"]["trajectory_plot"] = trajectory_plot_path

    metrics_plot_dir = outputs.get("metrics_plot_dir")
    if metrics_plot_dir:
        Path(metrics_plot_dir).mkdir(parents=True, exist_ok=True)
        visualizer.plot_metrics(analysis_results, save_dir=metrics_plot_dir)
        summary["outputs"]["metrics_plot_dir"] = metrics_plot_dir
    else:
        visualizer.plot_metrics(analysis_results)

    csv_dir = outputs.get("csv_dir")
    if csv_dir:
        export_to_csv(analysis_results, csv_dir)
        summary["outputs"]["csv_dir"] = csv_dir

    animation_path = outputs.get("animation_path")
    if animation_path:
        _ensure_parent_dir(animation_path)
        visualizer.create_animation(
            np.array(smooth_trajectory),
            analysis_results,
            title="Analisi della Traiettoria del Bilanciere",
            save_path=animation_path,
            fps=visualization.get("animation_fps", 10),
        )
        summary["outputs"]["animation"] = animation_path

    if foot_position is not None and squat_analysis is not None:
        squat_video_path = outputs.get("squat_analysis_video_path")
        if squat_video_path:
            _ensure_parent_dir(squat_video_path)
            visualizer.visualize_squat_analysis(
                input_video,
                smooth_trajectory,
                foot_position,
                squat_analysis,
                squat_video_path,
            )
            summary["outputs"]["squat_analysis_video"] = squat_video_path

    trajectory_video_path = outputs.get("trajectory_video_path")
    if trajectory_video_path:
        _ensure_parent_dir(trajectory_video_path)
        visualizer.plot_trajectory_on_video(
            smooth_trajectory,
            video_path=input_video,
            output_path=trajectory_video_path,
            draw_path=visualization.get("draw_path", True),
            path_history=visualization.get("path_history"),
            show_velocity=visualization.get("show_velocity", True),
            analysis_results=analysis_results,
        )
        summary["outputs"]["trajectory_video"] = trajectory_video_path


def _ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _summarize_trajectory_analysis(analysis_results):
    return {
        "max_velocity": float(analysis_results["max_velocity"]),
        "max_acceleration": float(analysis_results["max_acceleration"]),
        "avg_velocity": float(analysis_results["avg_velocity"]),
        "total_displacement": float(analysis_results["total_displacement"]),
        "total_path_length": float(analysis_results["total_path_length"]),
    }


def _summarize_squat_analysis(squat_analysis):
    return {
        "mean_deviation_px": float(squat_analysis["mean_deviation_px"]),
        "max_forward_deviation_px": float(squat_analysis["max_forward_deviation_px"]),
        "max_backward_deviation_px": float(squat_analysis["max_backward_deviation_px"]),
        "std_deviation_px": float(squat_analysis["std_deviation_px"]),
        "percent_in_safe_zone": float(squat_analysis["percent_in_safe_zone"]),
        "assessment": squat_analysis["assessment"],
    }
