"""
cache_images.py  -  OPTIONAL but recommended on CPU: resize every image ONCE and store it in cache/<size>/.

Why: BreaKHis images are 700x460 PNG. Without the cache, every epoch re-decodes and re-resizes them
(slow, and it runs on the same CPU that trains the network). With the cache, training reads small
ready-made images. The cached image is exactly what the training transform would produce, so
accuracy is unchanged. dataset.py picks the cache up automatically if it exists.

Run once per image size (all folds share the same images, so any split csv works):
    python cache_images.py --split-csv data/split_fold0.csv --img-size 160
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image

import config as C

RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR


def _one(job):
    src, dst, size = job
    if os.path.exists(dst):
        return 0
    img = Image.open(src).convert("RGB").resize((size, size), RESAMPLE)
    tmp = dst + ".tmp"
    img.save(tmp, format="PNG", compress_level=1)
    os.replace(tmp, dst)            # a half-written file can never be mistaken for a finished one
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--img-size", type=int, default=C.IMG_SIZE)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    df = pd.read_csv(args.split_csv).drop_duplicates("filename")
    out = C.CACHE_DIR / str(args.img_size)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, str(out / f), args.img_size) for p, f in zip(df.image_path, df.filename)]
    print(f"caching {len(jobs)} images at {args.img_size}x{args.img_size} with {args.workers} workers -> {out}")
    with ProcessPoolExecutor(args.workers) as ex:
        made = sum(ex.map(_one, jobs, chunksize=32))
    mb = sum(f.stat().st_size for f in out.glob("*.png")) / 1e6
    print(f"done: {made} new, {len(jobs) - made} already cached | cache size {mb:.0f} MB")


if __name__ == "__main__":
    main()
