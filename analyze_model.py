import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import Model


CLASS_NAMES = ["Down", "Neutral", "Up"]
FEATURE_COLUMNS = ["RSI", "MACD", "trend_score"]
DEFAULT_CSV = r"D:\full-stack2\data\labels\labeled_dataset.csv"
DEFAULT_RAW_DIR = r"D:\full-stack2\data\raw"


class CandlestickDataset(Dataset):
    def __init__(self, csv_path, split="test", kaggle_random_val=False):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if Path(row["image_path"]).exists()]

        if kaggle_random_val:
            try:
                from sklearn.model_selection import train_test_split
                _, rows = train_test_split(
                    rows,
                    test_size=0.2,
                    stratify=[row["label_id"] for row in rows],
                    random_state=42,
                )
            except ImportError as exc:
                raise RuntimeError("Install scikit-learn to use --kaggle-random-val") from exc
            self.rows = list(rows)
        else:
            self.rows = [row for row in rows if row.get("split") == split]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        features = [float(row.get(col) or 0.0) for col in FEATURE_COLUMNS]
        return (
            self.transform(image),
            torch.tensor(features, dtype=torch.float32),
            int(row["label_id"]),
            idx,
        )


def load_trained_model(checkpoint_path, device):
    model = Model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_metrics(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1

    report = {}
    f1_values = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        }

    accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(cm[row][col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


class ViTGradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        target_layer = self.model.vit.blocks[-1].norm1
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image, features, target_class=None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(image.to(self.device), features.to(self.device))
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        output[0, target_class].backward()

        activations = self.activations[:, 1:, :]
        gradients = self.gradients[:, 1:, :]
        weights = gradients.mean(dim=2, keepdim=True)
        cam = torch.relu((activations * weights).sum(dim=2)).squeeze(0)
        side = int(cam.numel() ** 0.5)
        cam = cam.reshape(side, side).detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def save_gradcam_overlay(image_path, heatmap, output_path, alpha=0.45):
    original = Image.open(image_path).convert("RGB").resize((224, 224))
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
    heatmap_arr = np.asarray(heatmap_img) / 255.0
    color_arr = (plt.get_cmap("jet")(heatmap_arr)[:, :, :3] * 255).astype(np.uint8)
    overlay = Image.blend(original, Image.fromarray(color_arr), alpha)
    overlay.save(output_path)


def read_raw_rows(raw_dir, ticker):
    raw_path = Path(raw_dir) / f"{ticker}.csv"
    if not raw_path.exists():
        return []
    with open(raw_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def detect_patterns(raw_rows, date):
    idx = next((i for i, row in enumerate(raw_rows) if row["Date"] == date), None)
    if idx is None:
        return ["Missing"], "Neutral"

    row = raw_rows[idx]
    open_price = float(row["Open"])
    high = float(row["High"])
    low = float(row["Low"])
    close = float(row["Close"])
    body = abs(close - open_price)
    candle_range = max(high - low, 1e-9)
    upper = high - max(open_price, close)
    lower = min(open_price, close) - low

    patterns = []
    signal = "Neutral"
    if body <= 0.10 * candle_range:
        patterns.append("Doji")
    if lower >= 2 * max(body, 1e-9) and upper <= max(body, 1e-9):
        patterns.append("Hammer")
        signal = "Up"

    if idx > 0:
        prev = raw_rows[idx - 1]
        prev_open = float(prev["Open"])
        prev_close = float(prev["Close"])
        bullish_engulfing = prev_close < prev_open and close > open_price and open_price <= prev_close and close >= prev_open
        bearish_engulfing = prev_close > prev_open and close < open_price and open_price >= prev_close and close <= prev_open
        if bullish_engulfing:
            patterns.append("Bullish Engulfing")
            signal = "Up"
        if bearish_engulfing:
            patterns.append("Bearish Engulfing")
            signal = "Down"

    return patterns or ["None"], signal


def write_report(metrics, pattern_summary, output_path):
    report = f"""# Candlestick ViT Stock Trend Prediction Report

## Labeling Strategy
Images are labeled from the future return in the prepared dataset. The project labels are mapped as `0=Down`, `1=Neutral`, and `2=Up`. If using `D:\\full-stack2\\data\\labels\\labeled_dataset.csv`, the label file includes a volatility-adjusted `trend_score`; the earlier config uses thresholds `Up >= 0.5`, `Down <= -0.5`, and values between them as `Neutral`.

## Model
The trained model in `best_model.pth` uses `vit_small_patch16_224` as the image encoder. Its ViT embedding is concatenated with three technical features: RSI, MACD, and MACD_signal, then classified into Down, Neutral, or Up.

## Test Evaluation
- Accuracy: {metrics['accuracy']:.4f}
- Macro F1: {metrics['macro_f1']:.4f}
- Confusion matrix order: Down, Neutral, Up
- Confusion matrix: `{metrics['confusion_matrix']}`

## XAI
Grad-CAM overlays were generated from the last transformer block. Bright regions show chart patches that most influenced the selected class score. Use the saved images in `outputs/xai` to discuss whether the model focuses on recent candle bodies, upper wicks, lower wicks, or volume/lower chart regions.

## Comparative Study
Classical rule checks were applied to the final candle of each chart window: Doji, Hammer, Bullish Engulfing, and Bearish Engulfing.

- Rows with any classical pattern: {pattern_summary['pattern_rows']}
- Classical signal agreement with ground truth: {pattern_summary['pattern_label_agreement']:.4f}
- Classical signal agreement with ViT prediction: {pattern_summary['pattern_prediction_agreement']:.4f}
- Pattern counts: {pattern_summary['pattern_counts']}

## Discussion
Deep learning can rediscover some classical candlestick ideas because candle bodies, wicks, gaps, and local reversals are visible image structures. However, ViT is not limited to named patterns: it can combine weak signals across many candles, technical indicators, and relative position in the chart. Classical rules are interpretable but rigid; the ViT may find softer or multi-candle structures that do not have a textbook name. The XAI overlays should be treated as evidence of focus, not as full proof of causality.
"""
    output_path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-xai", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--kaggle-random-val", action="store_true", help="Match Kaggle train_test_split validation.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    xai_dir = output_dir / "xai"
    output_dir.mkdir(exist_ok=True)
    xai_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CandlestickDataset(args.csv, args.split, args.kaggle_random_val)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = load_trained_model(args.checkpoint, device)

    y_true, y_pred, prediction_rows = [], [], []
    with torch.no_grad():
        for images, features, labels, indices in loader:
            logits = model(images.to(device), features.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            for row_idx, pred, prob in zip(indices.numpy().tolist(), preds.tolist(), probs.tolist()):
                row = dict(dataset.rows[row_idx])
                row["predicted_label"] = CLASS_NAMES[pred]
                row["confidence"] = max(prob)
                prediction_rows.append(row)

    metrics = compute_metrics(y_true, y_pred)
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")

    raw_cache = {}
    pattern_counts = Counter()
    pattern_rows = 0
    pattern_label_agree = 0
    pattern_pred_agree = 0
    pattern_output = []
    for row in prediction_rows:
        ticker = row["ticker"]
        raw_cache.setdefault(ticker, read_raw_rows(args.raw_dir, ticker))
        patterns, signal = detect_patterns(raw_cache[ticker], row["date"])
        pattern_counts.update(patterns)
        if patterns != ["None"] and patterns != ["Missing"]:
            pattern_rows += 1
            pattern_label_agree += int(signal == row["label"])
            pattern_pred_agree += int(signal == row["predicted_label"])
        pattern_output.append({**row, "patterns": "|".join(patterns), "pattern_signal": signal})

    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pattern_output[0].keys())
        writer.writeheader()
        writer.writerows(pattern_output)

    pattern_summary = {
        "pattern_rows": pattern_rows,
        "pattern_label_agreement": pattern_label_agree / pattern_rows if pattern_rows else 0.0,
        "pattern_prediction_agreement": pattern_pred_agree / pattern_rows if pattern_rows else 0.0,
        "pattern_counts": dict(pattern_counts),
    }
    (output_dir / "pattern_comparison.json").write_text(json.dumps(pattern_summary, indent=2), encoding="utf-8")

    gradcam = ViTGradCAM(model, device)
    for idx, row in enumerate(prediction_rows[:args.max_xai]):
        image = dataset.transform(Image.open(row["image_path"]).convert("RGB")).unsqueeze(0)
        features = torch.tensor([[float(row.get(col) or 0.0) for col in FEATURE_COLUMNS]], dtype=torch.float32)
        heatmap = gradcam.generate(image, features)
        out_name = f"{idx + 1:02d}_{row['ticker']}_{row['date']}_{row['predicted_label']}.png"
        save_gradcam_overlay(row["image_path"], heatmap, xai_dir / out_name)
    gradcam.close()

    write_report(metrics, pattern_summary, output_dir / "report.md")
    print(f"Done. Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
