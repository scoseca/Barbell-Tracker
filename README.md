# Barbell Tracker

Python pipeline for barbell trajectory tracking and squat analysis from video.

The project combines a YOLO detector, DeepSORT tracking, MediaPipe pose landmarks, trajectory analysis, and visual output generation.

## Features

- Detect and track the barbell across a video.
- Select the longest tracked trajectory as the main bar path.
- Smooth the trajectory and compute velocity, acceleration, displacement, and path length.
- Estimate the midfoot position with MediaPipe pose landmarks.
- Compare the barbell path against the midfoot for squat analysis.
- Generate plots, videos, animations, and CSV exports.

## Project Structure

```text
barbell-tracker/
+-- assets/
|   +-- demo.gif
|   +-- trajectory-example.png
|   +-- squat-analysis-frame.png
|   +-- pipeline-overview.png
+-- configs/
|   +-- pipeline.example.yaml
+-- src/
|   +-- barbell_tracker/
|   |   +-- cli.py
|   |   +-- config.py
|   |   +-- pipeline.py
|   |   +-- analysis/
|   |   +-- tracking/
|   |   +-- pose/
|   |   +-- visualization/
|   |   +-- export/
|   |   +-- utils/
|   +-- detection/
|       +-- legacy training and validation scripts
+-- tests/
+-- main.py
+-- requirements.txt
+-- README.md
```

Local-only folders such as `models/`, `outputs/`, `runs/`, large datasets, and raw videos are ignored by git.

## Visual Examples

The repository keeps a few lightweight visual examples under `assets/` so the README can show the expected output without committing full videos, datasets, model weights, or training runs.

![Pipeline overview](assets/pipeline-overview.png)

- `assets/demo.gif`: short trajectory animation ([open demo](assets/demo.gif)).
- `assets/trajectory-example.png`: static barbell trajectory plot.
- `assets/squat-analysis-frame.png`: annotated frame from the squat safe-zone video.
- `assets/pipeline-overview.png`: high-level pipeline diagram.

| Trajectory plot | Squat analysis frame |
| --- | --- |
| ![Trajectory example](assets/trajectory-example.png) | ![Squat analysis frame](assets/squat-analysis-frame.png) |

## Installation

Using Anaconda is recommended if you already developed the project that way:

```powershell
conda create -n barbell-tracker python=3.10
conda activate barbell-tracker
pip install -r requirements.txt
```

If you already have an exported conda environment file, you can use it instead:

```powershell
conda env create -f path\to\environment.yaml
conda activate barbell-tracker
```

## Models and Data

Create these local folders manually:

```text
models/
+-- mediapipe/
|   +-- pose_landmarker_heavy.task
+-- yolo/
    +-- barbell/
        +-- best.pt

sample_data/
+-- short-demo.mp4
```

Recommended paths match `configs/pipeline.example.yaml`:

- `models/yolo/barbell/best.pt`: your trained YOLO barbell detector.
- `models/mediapipe/pose_landmarker_heavy.task`: MediaPipe Pose Landmarker model.
- `sample_data/short-demo.mp4`: a short local test video.

Do not commit model weights, raw datasets, generated runs, or long videos.

## Configuration

Copy the example config and edit paths for your machine:

```powershell
Copy-Item configs\pipeline.example.yaml configs\pipeline.yaml
```

Important fields:

- `paths.input_video`: input video.
- `paths.output_dir`: base output directory.
- `models.yolo_model_path`: YOLO `.pt` weights.
- `models.pose_landmarker_path`: MediaPipe `.task` file.
- `runtime.device`: `cuda` or `cpu`.
- `runtime.confidence`: detection threshold.
- `analysis.squat.safe_zone_px`: allowed horizontal deviation from midfoot.
- `foot_detection.n_clusters`: KMeans clusters for stable foot position.

## Run

Run from the project root:

```powershell
py -3 main.py --config configs\pipeline.yaml
```

You can override key config values from the CLI:

```powershell
py -3 main.py `
  --config configs\pipeline.yaml `
  --input sample_data\short-demo.mp4 `
  --yolo_model_path models\yolo\barbell\best.pt `
  --pose_landmarker_path models\mediapipe\pose_landmarker_heavy.task `
  --device cuda `
  --confidence 0.5
```

## Example Outputs

With the example config, generated files are written under `outputs/demo/`:

```text
outputs/demo/
+-- barbell_position.png
+-- barbell_trajectory_animation.gif
+-- trajectory_video.mp4
+-- squat_analysis.mp4
+-- foot_position_clusters.png
+-- csv/
    +-- position_data.csv
    +-- velocity_data.csv
    +-- acceleration_data.csv
    +-- summary_stats.csv
```

Metric plots are also saved in the configured metrics directory with timestamped names.

## Metrics

The pipeline extracts:

- horizontal displacement from the estimated midfoot position;
- vertical barbell displacement;
- velocity and acceleration curves;
- total path length;
- percentage of frames inside the safe zone;
- trajectory visualization over the original video.

Example summary:

```text
valid trajectory points: 142
selected track ID: 3
estimated FPS: 30
average horizontal deviation: 24.6 px
max horizontal deviation: 71.3 px
time inside safe zone: 82.4%
```

Pixel-based distances depend on the input video resolution and camera framing, so compare them across videos only when the camera setup is consistent.

## Tests

Run the current unit tests with:

```powershell
py -3 -m unittest discover -s tests
```

The tests use synthetic data and do not require YOLO weights, MediaPipe models, or a GPU.
