"""
predict.py  -  STEP 3: run the trained classifier on every VALIDATION and TEST image
               (all four magnifications) and calibrate its confidence.

Why calibrate?  The escalation rule says "stop if confidence >= threshold".
That only means something if "90% confident" really means "right ~90% of the time".
We fit ONE number, a temperature T, on the validation set:  p = softmax(logits / T).
T is never fitted on test data.

Output: results/preds_<tag>.csv  (one row per image, with raw + calibrated probability)

Run:  python predict.py --checkpoint checkpoints/shared_seed0.pth --split-csv data/split_seed42.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as C
from dataset import ImageRows
from model import SharedMagClassifier


@torch.no_grad()
def get_logits(model, df, img_size, batch_size, workers, device):
    loader = DataLoader(ImageRows(df, img_size, False), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    model.eval()
    out = []
    for x, _, _ in loader:
        out.append(model(x.to(device)).float().cpu())
    return torch.cat(out)


def fit_temperature(logits, labels):
    """Find T > 0 minimising NLL of softmax(logits / T) on the validation set."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def ece(prob_malignant, labels, bins=10):
    """Expected calibration error on the predicted class's confidence."""
    pred = (prob_malignant >= 0.5).astype(int)
    conf = np.where(pred == 1, prob_malignant, 1 - prob_malignant)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tag = args.tag or args.checkpoint.replace("\\", "/").split("/")[-1].replace("shared_", "").replace(".pth", "")
    model = SharedMagClassifier(pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    df = pd.read_csv(args.split_csv)
    parts = []
    for split in ("validation", "test"):
        d = df[df.split == split].reset_index(drop=True)
        lg = get_logits(model, d, ck["img_size"], args.batch_size, args.workers, device)
        d["logit0"], d["logit1"] = lg[:, 0].numpy(), lg[:, 1].numpy()
        parts.append((split, d, lg))
        print(f"{split}: {len(d)} images")

    # ---- temperature from VALIDATION only ----------------------------------
    _, dval, lval = parts[0]
    yval = torch.tensor(dval["label"].values)
    T = fit_temperature(lval, yval)
    print(f"\nfitted temperature T = {T:.3f}  (T>1 means the raw model was over-confident)")

    out = []
    for split, d, lg in parts:
        d["p_raw"] = F.softmax(lg, 1)[:, 1].numpy()
        d["p_cal"] = F.softmax(lg / T, 1)[:, 1].numpy()
        y = d["label"].values
        print(f"{split:10s} ECE raw {ece(d['p_raw'].values, y):.4f} -> calibrated {ece(d['p_cal'].values, y):.4f}")
        out.append(d)
    res = pd.concat(out, ignore_index=True)

    f = C.RESULTS_DIR / f"preds_{tag}.csv"
    res.to_csv(f, index=False)
    json.dump({"temperature": T, "checkpoint": args.checkpoint}, open(C.RESULTS_DIR / f"temperature_{tag}.json", "w"))
    print("Saved:", f)


if __name__ == "__main__":
    main()
