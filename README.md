# Plant Disease Detection

This repository contains a YOLO-based computer vision project for plant disease and weed detection. It includes training outputs, inference scripts, a GPU verification script, and the dataset structure used for experimentation.

## Contents

- `weed_only_inference.py` - runs inference with a trained YOLO model and labels weed classes separately from crops.
- `gradcam_single_image.py` - generates a Grad-CAM style heatmap for a single image using a trained YOLO model.
- `verify_gpu.py` - checks whether PyTorch can access CUDA and a GPU device.
- `PDT_Training_Report.pdf` - project training/report document.

## Repository Layout

- `Datasets/` - dataset folders and annotations used for training and validation.
- `runs/` - YOLO training, validation, prediction, and output artifacts.
- `yolo11n.pt`, `yolov8n.pt`, `yolov8s.pt` - pretrained YOLO weights used for experiments.

## Requirements

The scripts are written in Python and rely on common computer vision libraries such as:

- `torch`
- `ultralytics`
- `opencv-python`
- `numpy`

Use the existing virtual environment if available, or install the dependencies in your own environment.

## Usage

### 1. Check GPU availability

Run:

```bash
python verify_gpu.py
```

### 2. Weed-only inference

Update the model path and input image in `weed_only_inference.py`, then run:

```bash
python weed_only_inference.py
```

The output image is written to the configured `runs/detect/weed_only_output/` folder.

### 3. Generate Grad-CAM visualization

Update the model path and image path in `gradcam_single_image.py`, then run:

```bash
python gradcam_single_image.py
```

The generated visualization is saved to the configured `runs/detect/gradcam_images/` folder.

## Notes

- Large generated folders such as `Datasets/`, `runs/`, and `venv/` are ignored by Git in this repository.
- The root model weight files `yolo11n.pt`, `yolov8n.pt`, and `yolov8s.pt` are tracked in Git so they can be published with the project.
- Some scripts use absolute Windows paths. If you move the project, update those paths before running the scripts.
