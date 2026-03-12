#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine images into a grid.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input image paths.")
    parser.add_argument("--out_path", required=True, help="Output image path.")
    parser.add_argument("--ncol", type=int, default=3, help="Number of columns.")
    parser.add_argument("--bg", default="white", help="Background color.")
    parser.add_argument("--pad", type=int, default=0, help="Padding between tiles (px).")
    parser.add_argument("--label_prefix", default=None, help="Prefix for panel labels, e.g. '(' for (a).")
    parser.add_argument("--label_suffix", default=None, help="Suffix for panel labels, e.g. ')' for (a).")
    parser.add_argument("--labels", nargs="+", default=None, help="Custom labels for each tile.")
    parser.add_argument("--labels_file", default=None, help="Path to newline-delimited custom labels.")
    parser.add_argument("--label_size", type=int, default=20, help="Label font size.")
    parser.add_argument("--label_margin", type=int, default=12, help="Label margin (px).")
    parser.add_argument("--label_band", type=int, default=0, help="Extra label band height under each tile (px).")
    parser.add_argument(
        "--label_pos",
        default="top-left",
        choices=["top-left", "bottom-center"],
        help="Label position within each tile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ncol <= 0:
        raise ValueError("--ncol must be positive.")

    inputs = [Path(p) for p in args.inputs]
    images = [Image.open(p).convert("RGB") for p in inputs]
    if not images:
        raise ValueError("No input images provided.")

    font = None
    font_scale = 1
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for cand in font_candidates:
        try:
            font = ImageFont.truetype(cand, args.label_size)
            break
        except OSError:
            continue
    if font is None:
        raise FileNotFoundError("DejaVu font not found at /usr/share/fonts/truetype/dejavu/")

    labels = None
    if args.labels_file:
        labels_path = Path(args.labels_file)
        labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
    elif args.labels:
        labels = list(args.labels)
    if labels is not None and len(labels) != len(images):
        raise ValueError(f"Label count ({len(labels)}) does not match image count ({len(images)}).")

    # Determine label band height, auto-expand if needed.
    label_band = max(args.label_band, 0)
    if labels is not None or args.label_prefix is not None or args.label_suffix is not None:
        if labels is None:
            labels_to_measure = [
                f"{args.label_prefix or ''}{chr(ord('a') + idx)}{args.label_suffix or ''}"
                for idx in range(len(images))
            ]
        else:
            labels_to_measure = labels
        tmp_img = Image.new("L", (1, 1), color=0)
        tmp_draw = ImageDraw.Draw(tmp_img)
        max_label_h = 0
        for label in labels_to_measure:
            bbox = tmp_draw.textbbox((0, 0), label, font=font)
            label_h = int((bbox[3] - bbox[1]) * font_scale)
            max_label_h = max(max_label_h, label_h)
        label_band = max(label_band, max_label_h + args.label_margin)

    tile_w = max(img.width for img in images)
    tile_h = max(img.height + label_band for img in images)
    ncol = args.ncol
    nrow = int(math.ceil(len(images) / ncol))
    pad = max(args.pad, 0)

    canvas_w = tile_w * ncol + pad * (ncol - 1)
    canvas_h = tile_h * nrow + pad * (nrow - 1)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=args.bg)
    draw = ImageDraw.Draw(canvas)

    label_prefix = args.label_prefix or ""
    label_suffix = args.label_suffix or ""
    for idx, img in enumerate(images):
        row = idx // ncol
        col = idx % ncol
        x0 = col * (tile_w + pad) + (tile_w - img.width) // 2
        y0 = row * (tile_h + pad) + (tile_h - img.height - label_band) // 2
        canvas.paste(img, (x0, y0))
        if labels is not None or args.label_prefix is not None or args.label_suffix is not None:
            if labels is not None:
                label = labels[idx]
            else:
                label = f"{label_prefix}{chr(ord('a') + idx)}{label_suffix}"
            bbox = draw.textbbox((0, 0), label, font=font)
            base_w = max(1, int(bbox[2] - bbox[0]))
            base_h = max(1, int(bbox[3] - bbox[1]))
            label_w = max(1, int(base_w * font_scale))
            label_h = max(1, int(base_h * font_scale))
            if args.label_pos == "bottom-center":
                lx = x0 + (img.width - label_w) // 2
                if label_band > 0:
                    ly = y0 + img.height + (label_band - label_h) // 2
                else:
                    ly = y0 + img.height - label_h - args.label_margin
            else:
                lx = x0 + args.label_margin
                ly = y0 + args.label_margin
            if font_scale == 1:
                draw.text((lx, ly), label, fill="black", font=font)
            else:
                tmp = Image.new("L", (base_w, base_h), color=0)
                tmp_draw = ImageDraw.Draw(tmp)
                tmp_draw.text((0, 0), label, fill=255, font=font)
                tmp = tmp.resize((label_w, label_h), resample=Image.BICUBIC)
                canvas.paste(Image.new("RGB", (label_w, label_h), "black"), (int(lx), int(ly)), tmp)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


if __name__ == "__main__":
    main()
