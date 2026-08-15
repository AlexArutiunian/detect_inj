#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import glob
import math
import numpy as np
import argparse

def show_image(img):
    """Try google.colab.patches.cv2_imshow; fallback to matplotlib if it fails."""
    try:
        from google.colab.patches import cv2_imshow  # may not exist outside Colab
        try:
            cv2_imshow(img)
            return
        except Exception:
            pass  # fall through to matplotlib
    except Exception:
        pass

    # Fallback: matplotlib (works everywhere)
    try:
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"[WARN] Failed to display image: {e}")

def parse_images_arg(images_args, input_dir):
    """
    Распарсить --images:
      - поддерживает пробелы и запятые
      - поддерживает glob-шаблоны (*.png)
      - относительные пути ищет внутри input_dir
    Сохраняет порядок, исключает дубли.
    """
    if not images_args:
        return []

    # разобрать по запятым и пробелам
    raw = []
    for token in images_args:
        raw.extend([p for p in token.split(",") if p.strip()])

    resolved = []
    seen = set()
    for p in raw:
        p = p.strip()
        # если это относительный путь и не найден, попробуем склеить с input_dir
        candidates = []
        if any(ch in p for ch in ["*", "?", "["]):
            # glob-шаблон
            matches = glob.glob(p)
            if not matches and not os.path.isabs(p):
                matches = glob.glob(os.path.join(input_dir, p))
            candidates = sorted(matches)
        else:
            if os.path.isabs(p) or os.path.exists(p):
                candidates = [p]
            else:
                cand = os.path.join(input_dir, p)
                candidates = [cand]

        for c in candidates:
            c = os.path.abspath(c)
            if c not in seen:
                resolved.append(c)
                seen.add(c)

    return resolved

def compute_grid(n_imgs, rows=None, cols=None):
    """Подобрать сетку под n_imgs, если rows/cols не заданы полностью."""
    if rows is not None and cols is not None:
        if rows * cols < n_imgs:
            raise ValueError(f"--rows*--cols={rows*cols} меньше числа картинок ({n_imgs})")
        return rows, cols
    if rows is not None:
        cols = math.ceil(n_imgs / rows)
        return rows, cols
    if cols is not None:
        rows = math.ceil(n_imgs / cols)
        return rows, cols
    # авто: почти квадратная сетка
    cols = math.ceil(math.sqrt(n_imgs))
    rows = math.ceil(n_imgs / cols)
    return rows, cols

def main():
    parser = argparse.ArgumentParser(description="Build collage from images.")
    parser.add_argument(
        "--input_dir", type=str, default="outputs_run/rf",
        help="Базовая папка: относительные имена из --images ищутся здесь (default: outputs_run/rf)"
    )
    parser.add_argument(
        "--images", nargs="*", default=["cm_test.png", "roc_test.png", "pr_test.png", "f1_dev.png"],
        help=("Список картинок (через пробел или запятую). "
              "Поддерживаются glob-шаблоны (*.png). "
              "Примеры: "
              "--images cm.png roc.png pr.png f1.png  "
              "--images cm.png,roc.png,pr.png,f1.png  "
              "--images '*.png'")
    )
    parser.add_argument(
        "--rows", type=int, default=None,
        help="Число строк в сетке. Если не задано — подберётся автоматически."
    )
    parser.add_argument(
        "--cols", type=int, default=None,
        help="Число столбцов в сетке. Если не задано — подберётся автоматически."
    )
    parser.add_argument(
        "--gap", type=int, default=6,
        help="Отступ (в пикселях) между ячейками и по краям (default: 6)"
    )
    parser.add_argument(
        "--line_color", type=str, default="30,30,30",
        help="Цвет фона/линий в формате 'B,G,R' (default: 30,30,30)"
    )
    parser.add_argument(
        "--output", type=str, default="collage.png",
        help="Куда сохранить коллаж (default: collage.png)"
    )
    parser.add_argument(
        "--no_show", action="store_true",
        help="Не открывать окно, только сохранить файл."
    )
    args = parser.parse_args()

    # разобрать цвет
    try:
        bgr = tuple(int(x) for x in args.line_color.split(","))
        if len(bgr) != 3:
            raise ValueError
        line_color = (bgr[0], bgr[1], bgr[2])
    except Exception:
        raise ValueError(f"Некорректный --line_color: {args.line_color}. Ожидается 'B,G,R', напр. '30,30,30'.")

    # получить список файлов
    image_paths = parse_images_arg(args.images, args.input_dir)
    if not image_paths:
        raise FileNotFoundError("Список --images пуст или не удалось найти ни одного файла.")

    # загрузить изображения
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot open file: {path}")
        # нормализация к 3-канальному BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        images.append(img)

    n = len(images)
    rows, cols = compute_grid(n, rows=args.rows, cols=args.cols)

    # выровнять размеры (не увеличиваем; приводим к минимальному h,w по всем)
    h_min = min(im.shape[0] for im in images)
    w_min = min(im.shape[1] for im in images)
    resized = [cv2.resize(im, (w_min, h_min), interpolation=cv2.INTER_AREA) for im in images]

    # итоговое полотно
    gap = int(args.gap)
    H = rows * h_min + (rows + 1) * gap
    W = cols * w_min + (cols + 1) * gap
    collage = np.full((H, W, 3), line_color, dtype=np.uint8)

    # раскладка по сетке (лишние ячейки остаются фоном)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            y = gap + r * (h_min + gap)
            x = gap + c * (w_min + gap)
            collage[y:y + h_min, x:x + w_min] = resized[idx]
            idx += 1

    # показать (если не запретили)
    if not args.no_show:
        show_image(collage)

    # сохранить
    ok = cv2.imwrite(args.output, collage)
    if not ok:
        raise RuntimeError(f"Failed to save collage to {args.output}")
    print(f"[OK] Collage saved to {args.output}")

if __name__ == "__main__":
    main()
