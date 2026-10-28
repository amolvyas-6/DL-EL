# Pothole model comparison

Train YOLO11s on bounding boxes and EfficientNetV2-S on road/pothole image labels. A zero-shot CLIP gate screens for non-road scenes (there are **no labeled non-road examples**). A variant of the custom model, served through local Ollama and the TypeSafe SDK, additionally rates the *apparent* severity of a pothole; this is not a physical measurement.

Every model reports one of three classes: `non-road`, `pothole` or `not-pothole`. YOLO has no non-road concept: it reports `pothole` if it finds a box, otherwise `not-pothole`.

## Setup

This project uses Python 3.12 and CUDA-enabled PyTorch wheels from the official cu128 index. An NVIDIA GPU with a working driver is required.

```sh
uv sync
uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'
```

Device placement:

| Model | Device |
|---|---|
| YOLO11s | CPU |
| Custom (CLIP gate + EfficientNetV2-S) | GPU |
| Custom with severity (Ollama) | GPU |

Pretrained weights for YOLO11s, OpenCLIP ViT-B/32 and EfficientNetV2-S download automatically on first use. For the severity variant, run Ollama locally; the model name is set by `OLLAMA_MODEL` in `pothole/severity.py` (override with the `SEVERITY_MODEL` environment variable) and must appear in `ollama list`. Requests ask Ollama to offload every layer to the GPU (`num_gpu: 999`), so the model has to fit in VRAM.

## Train and evaluate

```sh
uv run python train_yolo.py --epochs 50
uv run python train_custom.py --epochs 25
uv run python benchmark.py
uv run python app.py
```

YOLO resets persisted Ultralytics settings before training, uses `dataset-yolo/data.yaml`, and writes its run to `runs/detect/train/`. When training finishes, the best weights are copied to `artifacts/yolo/best.pt` and the Ultralytics plots to `artifacts/yolo_plots/`; the app and benchmark read the copy. Custom weights: `artifacts/custom/best.pt`, with its figures in `artifacts/custom_plots/` and the benchmark figures in `artifacts/benchmark_plots/`. The benchmark report is saved to `artifacts/benchmark.json` and includes false-positive/false-negative file paths.

The common evaluation uses all 143 normal-road and 193 pothole images in `dataset-custom/val` and maps YOLO's detections to image-level presence/absence. EfficientNet's decision threshold and early stopping use a **training-only** holdout; the validation images are never used for selection. A 95% pothole recall target is **not guaranteed**. CLIP is included in all custom pipeline timings and can cause false negatives. Benchmark timings use batch size 1 on the same images, with YOLO on the CPU and the custom pipeline on the GPU, so they are not a like-for-like hardware comparison. Non-road and severity accuracy cannot be measured without labeled examples.

## Demo app

- **Image · YOLO vs Custom** and **Image · Custom with severity**: upload any number of images (each is compressed to at most 1024 px, JPEG quality 85, before classification), then browse the results with the arrow buttons beside the image. The first tab shows each model's class, confidence (YOLO box score or softmax) and latency; the second shows the class and severity (`low`, `medium`, `high`, or `None` when the image is not a pothole), each with its probability.
- **Video · YOLO vs Custom**: plays the video at its own speed and shows the current frame's class, confidence and latency per model. If inference is slower than the video, frames are skipped to stay in real time.

Check syntax with `python -m compileall -q train_custom.py pothole app.py benchmark.py`.
