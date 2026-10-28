"""Frozen image-level comparison on custom validation images, batch size one."""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

from PIL import Image

from pothole.models import POTHOLE, CustomClassifier, YoloClassifier
from pothole.plots import plot_benchmark


class Benchmark:
    def __init__(self, dataset: Path, yolo: Path, custom: Path):
        self.dataset = dataset
        self.models = {"yolo": YoloClassifier(yolo), "custom": CustomClassifier(custom)}

    def run(self) -> dict:
        paths = [(path, label) for folder, label in (("normal_road", 0), ("potholes", 1))
                 for path in sorted((self.dataset / "val" / folder).glob("*.jpg"))]
        if not paths or not any(label for _, label in paths) or not any(not label for _, label in paths):
            raise ValueError("Both classes must be present in dataset-custom/val.")
        # Warm each model without counting initialization/cold cache.
        with Image.open(paths[0][0]) as image:
            warmup = image.convert("RGB")
        for model in self.models.values():
            for _ in range(3):
                model.predict(warmup)
        output = {}
        for name, model in self.models.items():
            counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            times = []
            mistakes = []
            for path, target in paths:
                start = perf_counter()
                with Image.open(path) as image:
                    result = model.predict(image.convert("RGB"))
                times.append((perf_counter() - start) * 1000)
                positive = result.label == POTHOLE
                key = "tp" if target and positive else "fn" if target else "fp" if positive else "tn"
                counts[key] += 1
                if key in ("fp", "fn"):
                    mistakes.append(str(path))
            tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
            output[name] = {
                **counts,
                "recall": tp / (tp + fn),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                "accuracy": (tp + tn) / len(paths),
                "median_ms": statistics.median(times),
                "p95_ms": sorted(times)[min(len(times) - 1, int(len(times) * 0.95))],
                "times_ms": times,
                "mistakes": mistakes,
            }
        output["method"] = ("batch=1, 3 warmups;"
                            "wall time includes decode/preprocess and complete pipelines; validation labels not used "
                            "for model selection. Non-road and severity unvalidated.")
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset-custom"))
    parser.add_argument("--yolo", type=Path, default=Path("artifacts/yolo/best.pt"))
    parser.add_argument("--custom", type=Path, default=Path("artifacts/custom/best.pt"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()
    results = Benchmark(args.data, args.yolo, args.custom, args.threads).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    plot_benchmark(results, args.output.parent / "benchmark_plots")
    print(json.dumps({name: {k: v for k, v in result.items() if k not in ("mistakes", "times_ms")}
                      for name, result in results.items() if isinstance(result, dict)}, indent=2))
    print(f"Full report: {args.output}; figures: {args.output.parent / 'benchmark_plots'}")
