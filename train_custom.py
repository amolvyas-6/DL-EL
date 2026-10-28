"""Fine-tune EfficientNetV2-S; calibrate on training-only holdout."""

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from torchvision.transforms import v2

from pothole.models import input_transform
from pothole.plots import plot_training


class RoadImages(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], training: bool):
        self.samples = samples
        self.transform = input_transform()
        if training:
            self.transform = v2.Compose([
                v2.Resize((432, 432)), v2.RandomResizedCrop(384, scale=(0.75, 1.0)),
                v2.RandomHorizontalFlip(), v2.ColorJitter(0.15, 0.15, 0.15),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label


class EfficientNetTrainer:
    def __init__(self, dataset: Path, output: Path):
        self.dataset = dataset
        self.output = output

    def run(self, epochs: int = 25) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check the NVIDIA driver and CUDA PyTorch installation.")
        torch.manual_seed(42)
        random.seed(42)
        train, calibration = [], []
        rng = random.Random(42)
        for label, folder in enumerate(("normal_road", "potholes")):
            paths = sorted((self.dataset / "train" / folder).glob("*.jpg"))
            if not paths:
                raise FileNotFoundError(self.dataset / "train" / folder)
            rng.shuffle(paths)
            pivot = max(1, int(0.15 * len(paths)))
            calibration += [(path, label) for path in paths[:pivot]]
            train += [(path, label) for path in paths[pivot:]]
        loader = DataLoader(RoadImages(train, True), batch_size=256, shuffle=True, num_workers=4)
        holdout = DataLoader(RoadImages(calibration, False), batch_size=256, num_workers=4)
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        self.output.mkdir(parents=True, exist_ok=True)
        best, patience = float("inf"), 0
        history = {k: [] for k in ("train_loss", "holdout_loss", "holdout_accuracy",
                                   "holdout_recall", "holdout_precision")}
        history["best_epoch"] = 1
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for images, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images.cuda()), targets.cuda())
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            model.eval()
            losses, predicted, actual = [], [], []
            with torch.inference_mode():
                for images, targets in holdout:
                    logits = model(images.cuda())
                    losses.append(criterion(logits, targets.cuda()).item())
                    predicted += logits.argmax(-1).cpu().tolist()
                    actual += targets.tolist()
            current = sum(losses) / len(losses)
            tp = sum(p == 1 and a == 1 for p, a in zip(predicted, actual))
            fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual))
            history["train_loss"].append(sum(train_losses) / len(train_losses))
            history["holdout_loss"].append(current)
            history["holdout_accuracy"].append(sum(p == a for p, a in zip(predicted, actual)) / len(actual))
            history["holdout_recall"].append(tp / sum(actual))
            history["holdout_precision"].append(tp / (tp + fp) if tp + fp else 0.0)
            print(f"epoch {epoch + 1}: train loss={history['train_loss'][-1]:.4f} "
                  f"calibration loss={current:.4f}", flush=True)
            if current < best:
                best, patience = current, 0
                history["best_epoch"] = epoch + 1
                torch.save(model.state_dict(), self.output / "best_model.pt")
            else:
                patience += 1
                if patience >= 5:
                    break
        model.load_state_dict(torch.load(self.output / "best_model.pt", weights_only=True))
        model.eval()
        scores, labels = [], []
        with torch.inference_mode():
            for images, targets in holdout:
                scores.extend(model(images.cuda()).softmax(-1)[:, 1].cpu().tolist())
                labels.extend(targets.tolist())
        # Highest F1 among operating points with >=95% recall on the TRAIN-derived holdout.
        candidates = sorted(set(scores) | {0.0})
        viable = []
        for cutoff in candidates:
            tp = sum(s >= cutoff and y == 1 for s, y in zip(scores, labels))
            fp = sum(s >= cutoff and y == 0 for s, y in zip(scores, labels))
            positives = sum(labels)
            if tp / positives >= 0.95:
                viable.append((2 * tp / (positives + tp + fp), cutoff))
        threshold = max(viable)[1]
        torch.save({"model": model.state_dict(), "threshold": threshold}, self.output / "best.pt")
        history.update(scores=scores, labels=labels, threshold=threshold)
        (self.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        plot_training(history, self.output.parent / "custom_plots")
        print(f"Saved checkpoint; training-only calibrated threshold={threshold:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset-custom"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/custom"))
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    EfficientNetTrainer(args.data, args.output).run(args.epochs)
