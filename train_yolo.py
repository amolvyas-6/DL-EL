"""Train YOLO11s on the provided bounding-box annotations."""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO, settings


class YoloTrainer:
    def __init__(self, data: Path, output: Path = Path("artifacts/yolo_plots"),
                 weights: Path = Path("artifacts/yolo/best.pt")):
        self.data = data
        self.output = output
        self.weights = weights

    def run(self, epochs: int = 50) -> Path:
        settings.reset()  # same as `yolo settings reset`; clear persisted overrides first
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check the NVIDIA driver and CUDA PyTorch installation.")
        model = YOLO("yolo11s.pt")
        model.train(data=str(self.data.resolve()), epochs=epochs, imgsz=320,
                    device=0, seed=42, name="train", exist_ok=True)
        print(f"Best checkpoint: {model.trainer.best}")
        metrics = model.val(data=str(self.data.resolve()), split="test", imgsz=320, device=0)
        print("YOLO test detection metrics:", metrics.results_dict)
        self.export_plots(Path(model.trainer.save_dir), self.output / "train")
        self.export_plots(Path(metrics.save_dir), self.output / "test")
        self.weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model.trainer.best, self.weights)
        print(f"Plots copied to {self.output}; weights copied to {self.weights}")
        return self.weights

    @staticmethod
    def export_plots(source: Path, destination: Path) -> None:
        """Copy Ultralytics' figures and logs (not the weights) so every plot lives under artifacts/."""
        destination.mkdir(parents=True, exist_ok=True)
        for file in source.iterdir():
            if file.suffix in (".png", ".jpg", ".csv", ".yaml"):
                shutil.copy2(file, destination / file.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset-yolo/data.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    YoloTrainer(args.data).run(args.epochs)
