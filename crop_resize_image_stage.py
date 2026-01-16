#!/usr/bin/env python3
"""Prepare two-resolution image sets with consistent FOV.

Per input image (original resolution):
1) Stage-1 (low-res): resize to 1/N of the original size (integer floor), then center-crop minimally
   so that width and height are multiples of `--multiple` (default: 14). Save to `--stage1-out`.

2) Stage-2 (high-res): generate an image whose size is exactly 2x the Stage-1 *pre-crop* size,
   then apply a crop box that is exactly 2x the Stage-1 crop box. This guarantees Stage-2 has
   the same field-of-view as Stage-1 but with 2x linear resolution. Save to `--stage2-out`.

Why this works:
- Stage-1 is: I1 = crop(resize(I0, s1), box1), s1 = 1/N
- Stage-2 is: I2 = crop(resize(I0, s2), box2), s2 = 2*s1, box2 = 2*box1

Example:
  python crop_resize_image.py \
    --images ./images \
    --stage1-out ./images_stage1_half_crop14 \
    --stage2-out ./images_stage2_2x_stage1 

Notes:
- Cropping is centered and minimal (only removes the smallest border necessary).
- EXIF orientation is respected (images are transposed before processing).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageOps


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def _iter_images(images_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p
    else:
        for p in images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _unit(v):
    import numpy as np

    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v
    return v / n


def _center_crop_to_multiple(w: int, h: int, multiple: int) -> Tuple[int, int, int, int]:
    """Return (left, top, right, bottom) crop box with minimal centered crop.

    Ensures output width/height are multiples of `multiple`.
    """
    if multiple <= 0:
        raise ValueError("multiple must be > 0")

    new_w = w - (w % multiple)
    new_h = h - (h % multiple)

    if new_w <= 0 or new_h <= 0:
        raise ValueError(
            f"Image too small to crop to a multiple of {multiple}: {w}x{h}."
        )

    dx = w - new_w
    dy = h - new_h

    left = dx // 2
    right = w - (dx - left)  # distribute odd pixel to the left side

    top = dy // 2
    bottom = h - (dy - top)

    return left, top, right, bottom


def _scale_box(box: Tuple[int, int, int, int], k: int) -> Tuple[int, int, int, int]:
    l, t, r, b = box
    return l * k, t * k, r * k, b * k


def _save_image(img: Image.Image, out_path: Path) -> None:
    _ensure_dir(out_path.parent)

    ext = out_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        img.save(out_path, quality=95, subsampling=0, optimize=True)
    elif ext == ".png":
        img.save(out_path, optimize=True)
    else:
        img.save(out_path)


def process_one(
    in_path: Path,
    images_root: Path,
    stage1_out_root: Path,
    stage2_out_root: Path,
    stage1_downscale_n: int,
    multiple: int,
) -> Tuple[bool, str]:
    rel = in_path.relative_to(images_root)
    out1 = stage1_out_root / rel
    out2 = stage2_out_root / rel

    try:
        with Image.open(in_path) as im0:
            im = ImageOps.exif_transpose(im0)
            w0, h0 = im.size

            # Stage-1 resize: 1/N of original (integer floor).
            n = int(stage1_downscale_n)
            if n <= 0:
                raise ValueError("stage1_downscale_n must be >= 1")
            w1 = max(1, w0 // n)
            h1 = max(1, h0 // n)
            im1_base = im.resize((w1, h1), resample=Image.Resampling.LANCZOS)

            # Stage-1 crop to multiple.
            box1 = _center_crop_to_multiple(w1, h1, multiple)
            im1 = im1_base.crop(box1)
            _save_image(im1, out1)

            # Stage-2: 2x Stage-1 pre-crop size, crop box scaled by 2.
            k = n
            w2 = w1 * k
            h2 = h1 * k
            im2_base = im.resize((w2, h2), resample=Image.Resampling.LANCZOS)
            box2 = _scale_box(box1, k)
            im2 = im2_base.crop(box2)
            _save_image(im2, out2)

            msg = (
                f"OK  {rel} | orig {w0}x{h0} -> stage1_base {w1}x{h1} crop {im1.size[0]}x{im1.size[1]} "
                f"-> stage2_base {w2}x{h2} crop {im2.size[0]}x{im2.size[1]}"
            )
            return True, msg

    except Exception as e:
        return False, f"ERR {rel} | {type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Stage-1 and Stage-2 image sets.")
    p.add_argument("--images", required=True, type=Path, help="Input images directory")
    p.add_argument(
        "--stage1-out",
        required=True,
        type=Path,
        help="Output directory for Stage-1 images (half-res then crop-to-multiple).",
    )
    p.add_argument(
        "--stage2-out",
        required=True,
        type=Path,
        help="Output directory for Stage-2 images (2x Stage-1 linear resolution; consistent FOV).",
    )
    p.add_argument(
        "--stage1-downscale-n",
        default=2,
        type=int,
        help="Stage-1 downscale factor N (Stage-1 base size = original size / N). Default: 2 (half-res).",
    )
    p.add_argument(
        "--multiple",
        default=14,
        type=int,
        help="Stage-1 crop so that width and height are multiples of this value (default: 14).",
    )
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument("--quiet", action="store_true", help="Suppress per-file logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    images_dir: Path = args.images
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Input directory not found: {images_dir}")

    _ensure_dir(args.stage1_out)
    _ensure_dir(args.stage2_out)

    files = list(_iter_images(images_dir, args.recursive))
    if not files:
        raise SystemExit(
            f"No images found in {images_dir} (supported: {sorted(SUPPORTED_EXTS)})"
        )

    ok = 0
    err = 0
    for f in files:
        success, msg = process_one(
            f,
            images_root=images_dir,
            stage1_out_root=args.stage1_out,
            stage2_out_root=args.stage2_out,
            stage1_downscale_n=args.stage1_downscale_n,
            multiple=args.multiple,
        )
        if success:
            ok += 1
        else:
            err += 1
        if not args.quiet:
            print(msg)

    print(f"Done. OK={ok} ERR={err} TOTAL={ok + err}")


if __name__ == "__main__":
    # Work around PIL's large-image protection if needed.
    Image.MAX_IMAGE_PIXELS = None
    main()
