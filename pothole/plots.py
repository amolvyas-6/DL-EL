"""Publication-style figures for the custom model's training and for the benchmark.

Run `uv run python -m pothole.plots` to rebuild the benchmark figures from artifacts/benchmark.json.
Every figure is saved as a 300 dpi PNG and a vector PDF.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"yolo": "#c2410c", "custom": "#1d4ed8"}
NAMES = {"yolo": "YOLO11s", "custom": "CLIP + EfficientNetV2-S"}
METRICS = ("recall", "precision", "f1", "accuracy")

STYLE = {
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100, "savefig.bbox": "tight",
}


def _save(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=300)
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)


def _pr_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores)
    hits = labels[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(1 - hits)
    return tp / hits.sum(), tp / (tp + fp), scores[order]


# ---------------------------------------------------------------- custom model training

def plot_training(history: dict, folder: Path) -> None:
    """`history` has per-epoch lists plus the holdout `scores`/`labels` and the chosen `threshold`."""
    with plt.rc_context(STYLE):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        best = history["best_epoch"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, history["train_loss"], "o-", label="Training loss", color="#1d4ed8")
        axes[0].plot(epochs, history["holdout_loss"], "s-", label="Holdout loss", color="#c2410c")
        axes[0].axvline(best, ls="--", color="gray", label=f"Best epoch ({best})")
        axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
        axes[0].legend()
        for key, label in (("accuracy", "Accuracy"), ("recall", "Recall"), ("precision", "Precision")):
            axes[1].plot(epochs, history[f"holdout_{key}"], "o-", label=label)
        axes[1].axvline(best, ls="--", color="gray")
        axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1.02), title="Holdout metrics (argmax decision)")
        axes[1].legend(loc="lower right")
        for ax in axes:
            ax.xaxis.get_major_locator().set_params(integer=True)
        fig.suptitle("EfficientNetV2-S fine-tuning", fontweight="bold")
        fig.tight_layout()
        _save(fig, folder, "training_curves")

        scores, labels = np.array(history["scores"]), np.array(history["labels"])
        recall, precision, cutoffs = _pr_curve(scores, labels)
        threshold = history["threshold"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(recall, precision, color="#1d4ed8")
        index = int(np.searchsorted(-cutoffs, -threshold, side="right")) - 1
        axes[0].plot(recall[index], precision[index], "o", color="#c2410c", ms=9,
                     label=f"Chosen threshold = {threshold:.3f}")
        axes[0].axvline(0.95, ls=":", color="gray", label="95% recall target")
        axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.02),
                    title="Precision–recall (training holdout)")
        axes[0].legend(loc="lower left")
        grid = np.linspace(0, 1, 201)
        tp = np.array([((scores >= t) & (labels == 1)).sum() for t in grid])
        fp = np.array([((scores >= t) & (labels == 0)).sum() for t in grid])
        positives = labels.sum()
        axes[1].plot(grid, tp / positives, label="Recall", color="#1d4ed8")
        axes[1].plot(grid, 2 * tp / np.maximum(positives + tp + fp, 1), label="F1", color="#2f6f4f")
        axes[1].plot(grid, tp / np.maximum(tp + fp, 1), label="Precision", color="#c2410c")
        axes[1].axvline(threshold, ls="--", color="gray", label="Chosen threshold")
        axes[1].set(xlabel="Decision threshold", ylabel="Score", ylim=(0, 1.02), title="Threshold calibration")
        axes[1].legend(loc="lower left")
        fig.tight_layout()
        _save(fig, folder, "calibration")


# ---------------------------------------------------------------- benchmark

def plot_benchmark(results: dict, folder: Path) -> None:
    models = [name for name in ("yolo", "custom") if name in results]
    with plt.rc_context(STYLE):
        # Classification metrics.
        fig, ax = plt.subplots(figsize=(7, 4.2))
        width = 0.8 / len(models)
        x = np.arange(len(METRICS))
        for i, name in enumerate(models):
            values = [results[name][m] for m in METRICS]
            bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width,
                          label=NAMES[name], color=COLORS[name])
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
        ax.set(xticks=x, xticklabels=[m.capitalize() if m != "f1" else "F1" for m in METRICS],
               ylim=(0, 1.1), ylabel="Score", title="Image-level pothole detection (validation set)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=len(models), frameon=False)
        ax.set_title("Image-level pothole detection (validation set)", pad=40)
        _save(fig, folder, "benchmark_metrics")

        # Confusion matrices.
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8))
        for ax, name in zip(np.atleast_1d(axes), models):
            r = results[name]
            matrix = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
            ax.imshow(matrix / matrix.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for (row, col), count in np.ndenumerate(matrix):
                share = count / matrix[row].sum()
                ax.text(col, row, f"{count}\n({share:.1%})", ha="center", va="center",
                        color="white" if share > 0.5 else "black")
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal road", "Pothole"],
                   yticklabels=["Normal road", "Pothole"], xlabel="Predicted", ylabel="Actual",
                   title=NAMES[name])
            ax.grid(False)
        fig.suptitle("Confusion matrices (rows normalised)", fontweight="bold")
        fig.tight_layout()
        _save(fig, folder, "benchmark_confusion")

        # Latency: median and p95 bars, plus distributions when per-image times were recorded.
        has_times = all("times_ms" in results[n] for n in models)
        fig, axes = plt.subplots(1, 2 if has_times else 1, figsize=(10 if has_times else 5.5, 4.2))
        axes = np.atleast_1d(axes)
        x = np.arange(2)
        for i, name in enumerate(models):
            values = [results[name]["median_ms"], results[name]["p95_ms"]]
            bars = axes[0].bar(x + (i - (len(models) - 1) / 2) * width, values, width,
                               label=NAMES[name], color=COLORS[name])
            axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
        axes[0].set(xticks=x, xticklabels=["Median", "95th percentile"], ylabel="Latency per image (ms)",
                    title="Latency (batch size 1)")
        axes[0].grid(axis="x", visible=False)
        axes[0].legend()
        if has_times:
            parts = axes[1].violinplot([results[n]["times_ms"] for n in models], showmedians=True)
            for body, name in zip(parts["bodies"], models):
                body.set_facecolor(COLORS[name])
                body.set_alpha(0.6)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                parts[key].set_color("black")
            axes[1].set(xticks=[1, 2][:len(models)], xticklabels=[NAMES[n] for n in models],
                        ylabel="Latency per image (ms)", title="Latency distribution")
            axes[1].tick_params(axis="x", labelsize=9)
        fig.tight_layout()
        _save(fig, folder, "benchmark_latency")

        # Accuracy against speed.
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for name in models:
            ax.scatter(results[name]["median_ms"], results[name]["f1"], s=140, color=COLORS[name],
                       label=NAMES[name], zorder=3)
        ax.set(xlabel="Median latency (ms)", ylabel="F1", ylim=(0, 1.05), xlim=(0, None),
               title="Accuracy versus speed")
        ax.legend(loc="lower right")
        _save(fig, folder, "benchmark_tradeoff")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark_plots"))
    args = parser.parse_args()
    plot_benchmark(json.loads(args.benchmark.read_text()), args.output)
    print(f"Figures written to {args.output}")
