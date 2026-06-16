import argparse

from src.barbell_tracker.config import apply_config_overrides, load_pipeline_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run the barbell tracking pipeline.")
    parser.add_argument(
        "--config",
        default="configs/pipeline.example.yaml",
        help="Path to a pipeline YAML configuration file.",
    )
    parser.add_argument("--input", dest="input_video", help="Video path.")
    parser.add_argument("--yolo_model_path", help="YOLOv8 model path.")
    parser.add_argument("--pose_landmarker_path", help="MediaPipe Pose Landmarker model path.")
    parser.add_argument(
        "--confidence",
        type=float,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use for inference.",
    )
    parser.add_argument("--output_dir", help="Directory for pipeline outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_pipeline_config(args.config)
    config = apply_config_overrides(
        config,
        {
            "paths.input_video": args.input_video,
            "paths.output_dir": args.output_dir,
            "models.yolo_model_path": args.yolo_model_path,
            "models.pose_landmarker_path": args.pose_landmarker_path,
            "runtime.confidence": args.confidence,
            "runtime.device": args.device,
        },
    )

    from src.barbell_tracker.pipeline import run_pipeline

    summary = run_pipeline(config)
    print_summary(summary)


def print_summary(summary):
    print("\n----- PIPELINE SUMMARY -----")
    print(f"Status: {summary['status']}")
    print(f"Input video: {summary['input_video']}")
    print(f"Tracked objects: {summary['tracked_objects']}")

    if summary.get("track_id") is not None:
        print(f"Selected track ID: {summary['track_id']}")
        print(f"Trajectory points: {summary['trajectory_points']}")
        print(f"FPS: {summary['fps']:.2f}")

    trajectory = summary.get("trajectory_analysis")
    if trajectory:
        print("Trajectory:")
        print(f"  Max velocity: {trajectory['max_velocity']:.2f} px/s")
        print(f"  Max acceleration: {trajectory['max_acceleration']:.2f} px/s^2")
        print(f"  Avg velocity: {trajectory['avg_velocity']:.2f} px/s")
        print(f"  Total displacement: {trajectory['total_displacement']:.2f} px")
        print(f"  Total path length: {trajectory['total_path_length']:.2f} px")

    if summary.get("foot_position") is not None:
        print(f"Foot position: {summary['foot_position']}")

    squat = summary.get("squat_analysis")
    if squat:
        print("Squat:")
        print(f"  Mean deviation: {squat['mean_deviation_px']:.2f} px")
        print(f"  Safe zone time: {squat['percent_in_safe_zone']:.2f}%")
        print(f"  Assessment: {squat['assessment']}")

    if summary.get("outputs"):
        print("Outputs:")
        for name, path in summary["outputs"].items():
            print(f"  {name}: {path}")

    if summary.get("warnings"):
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"  {warning}")
    print("----------------------------\n")
