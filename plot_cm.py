#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # рисуем без GUI
import matplotlib.pyplot as plt


def load_cm(metrics_path: str) -> np.ndarray:
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    cm = np.array(m.get("confusion_matrix", []), dtype=np.int64)
    if cm.shape != (2, 2):
        raise ValueError(f"В файле {metrics_path} не найдена 2x2 confusion_matrix.")
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    labels=("No Injury (0)", "Injury (1)"),
    title="Confusion Matrix",
    out_path="cm.png",
):
    # нормировка по строкам (по истинному классу) — как на примере
    row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
    row_sum[row_sum == 0] = 1.0
    cm_norm = cm / row_sum

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Proportion per true class", rotation=270, labelpad=14)

    # подписи
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=[labels[0], labels[1]])
    ax.set_yticks([0, 1], labels=[labels[0], labels[1]])

    # линии сетки для аккуратности
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # подписи в ячейках: число и доля
    thresh = 0.5
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            frac = cm_norm[i, j]
            txt_color = "white" if frac > thresh else "black"
            ax.text(
                j, i,
                f"{count}\n{frac:.2f}",
                ha="center", va="center",
                fontsize=12, color=txt_color
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser("Draw confusion matrix image from saved metrics_*.json")
    p.add_argument("--metrics", required=True, help="Путь к metrics_dev.json или metrics_test.json")
    p.add_argument("--out", default="cm.png", help="Куда сохранить картинку PNG")
    p.add_argument("--title", default="Confusion Matrix", help="Заголовок картинки")
    p.add_argument(
        "--labels",
        default="No Injury (0),Injury (1)",
        help="Подписи классов через запятую: 'label0,label1'",
    )
    args = p.parse_args()

    labels = tuple(x.strip() for x in args.labels.split(","))
    if len(labels) != 2:
        raise ValueError("--labels должно содержать ровно 2 подписи через запятую")

    cm = load_cm(args.metrics)
    plot_confusion_matrix(cm, labels=labels, title=args.title, out_path=args.out)


if __name__ == "__main__":
    main()
