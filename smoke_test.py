"""
smoke_test.py  -  run this ONCE on your laptop before touching real data.

It creates a tiny FAKE BreaKHis-style folder (64x64 noise images, 30 fake patients),
then runs the whole pipeline: prepare_data -> train -> predict -> escalate.
It needs no internet and no pretrained weights, and takes a few minutes on CPU.
If it prints SMOKE TEST PASSED, every script and every import works on your machine.

The numbers it produces are meaningless (fake data). Afterwards you may delete:
    smoke_data/   data/split_seed999.csv   checkpoints/shared_smoke.pth   results/*smoke*   cache/64

Run:  python smoke_test.py
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import config as C

ROOT = C.PROJECT_DIR / "smoke_data"


def make_fake_dataset():
    rng = np.random.default_rng(0)
    pats = [("B", "A", f"{1000 + i}") for i in range(10)] + [("M", "DC", f"{2000 + i}") for i in range(20)]
    for cls, sub, slide in pats:
        cname = "benign" if cls == "B" else "malignant"
        base = np.array([200, 120, 170] if cls == "B" else [150, 80, 140], dtype=float)
        for mag in (40, 100, 200, 400):
            d = ROOT / f"{mag}X" / cname
            d.mkdir(parents=True, exist_ok=True)
            for seq in range(1, 7):
                img = np.clip(base + rng.normal(0, 35, (64, 64, 3)), 0, 255).astype("uint8")
                Image.fromarray(img).save(d / f"SOB_{cls}_{sub}-14-{slide}-{mag}-{seq:03d}.png")


def run(cmd, env=None):
    print("\n>>>", " ".join(cmd))
    r = subprocess.run([sys.executable] + cmd, cwd=C.PROJECT_DIR, env=env)
    if r.returncode != 0:
        raise SystemExit(f"\nSMOKE TEST FAILED at: {' '.join(cmd)}")


if __name__ == "__main__":
    make_fake_dataset()
    env = dict(os.environ, BREAKHIS_ROOT=str(ROOT))
    run(["prepare_data.py", "--seed", "999"], env=env)
    run(["cache_images.py", "--split-csv", "data/split_seed999.csv", "--img-size", "64", "--workers", "2"])
    run(["train.py", "--split-csv", "data/split_seed999.csv", "--tag", "smoke", "--no-pretrained",
         "--img-size", "64", "--epochs", "2", "--batch-size", "16", "--freeze-stages", "2", "--workers", "2"])
    run(["predict.py", "--checkpoint", "checkpoints/shared_smoke.pth", "--split-csv", "data/split_seed999.csv"])
    run(["escalate.py", "--preds", "results/preds_smoke.csv"])
    print("\nSMOKE TEST PASSED - the pipeline runs end to end on your machine.")
