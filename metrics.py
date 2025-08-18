#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
from google.colab.patches import cv2_imshow

def main():
    parser = argparse.ArgumentParser(description="Build 2x2 collage from metrics images.")
    parser.add_argument(
        "--input_dir", type=str, default="outputs_run/rf",
        help="Directory where metric images are stored (default: outputs_run/rf)"
    )
    parser.add_argument(
        "--output", type=str, default="collage_2x2.png",
        help="Where to save the collage (default: collage_2x2.png)"
    )
    args = parser.parse_args()

    metrics = ["cm_test.png", "roc_test.png", "pr_test.png", "f1_dev.png"]

    # Collage parameters
    rows, cols = 2, 2
    gap = 6
    line_color = (30, 30, 30)  # dark gray

    # Load images
    images = []
    for metric in metrics:
        path = os.path.join(args.input_dir, metric)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot open file: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        images.append(img)

    # Resize to same shape
    h_min = min(im.shape[0] for im in images)
    w_min = min(im.shape[1] for im in images)
    resized = [cv2.resize(im, (w_min, h_min), interpolation=cv2.INTER_AREA) for im in images]

    # Final canvas with margins
    H = rows * h_min + (rows + 1) * gap
    W = cols * w_min + (cols + 1) * gap
    collage = np.full((H, W, 3), line_color, dtype=np.uint8)

    # Place 2x2
    idx = 0
    for r in range(rows):
        for c in range(cols):
            y = gap + r * (h_min + gap)
            x = gap + c * (w_min + gap)
            collage[y:y+h_min, x:x+w_min] = resized[idx]
            idx += 1

    # Show (for Colab)
    cv2_imshow(collage)

    # Save
    cv2.imwrite(args.output, collage)
    print(f"[OK] Collage saved to {args.output}")

if __name__ == "__main__":
    main()
