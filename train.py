"""
train.py  -  STEP 2: train the shared single-image classifier.

Training data  : every TRAIN-split image, all four magnifications mixed together.
Early stopping : on VALIDATION loss (test set is never touched here).
Class imbalance: class-weighted cross-entropy, w_c = N / (K * n_c).

Run:  python train.py --split-csv data/split_seed42.csv --seed 0
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from dataset import ImageRows
from model import SharedMagClassifier


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--seed", type=int, default=0, help="training seed (weights init + batch order)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=C.IMG_SIZE)
    ap.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    ap.add_argument("--freeze-stages", type=int, default=0, help="freeze stem + first N stages (0-4); 2 is a good CPU setting")
    ap.add_argument("--no-pretrained", action="store_true", help="only for offline smoke tests")
    ap.add_argument("--tag", default=None, help="name of this run (default: seed<k>)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"seed{args.seed}"
    print(f"device={device} | run={tag}")

    df = pd.read_csv(args.split_csv)
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    print(f"train images={len(tr)} | val images={len(va)} "
          f"| train patients={tr.patient_id.nunique()} | val patients={va.patient_id.nunique()}")

    # ---- class weights from TRAIN counts only ------------------------------
    counts = tr["label"].value_counts().sort_index().values.astype(float)
    weights = counts.sum() / (len(counts) * counts)
    print("class counts", counts.astype(int).tolist(), "-> weights", weights.round(4).tolist())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    val_criterion = nn.CrossEntropyLoss()  # UNweighted val loss: a clean number to compare epochs

    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(ImageRows(tr, args.img_size, True), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, generator=g, drop_last=len(tr) > args.batch_size)
    vl = DataLoader(ImageRows(va, args.img_size, False), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers)

    model = SharedMagClassifier(pretrained=not args.no_pretrained)
    model.freeze_early(args.freeze_stages)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable):,} of {sum(p.numel() for p in model.parameters()):,}"
          f" (freeze_stages={args.freeze_stages})")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = device.type == "cuda"
    try:                                   # new torch versions
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):    # older torch versions
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt = C.CKPT_DIR / f"shared_{tag}.pth"
    best, bad, hist = float("inf"), 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y, _ in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            loss_sum += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
        sched.step()
        v_loss, v_acc = evaluate(model, vl, val_criterion, device)
        row = dict(epoch=ep, train_loss=loss_sum / n, train_acc=correct / n,
                   val_loss=v_loss, val_acc=v_acc, secs=round(time.time() - t0, 1))
        hist.append(row)
        flag = ""
        if v_loss < best:
            best, bad = v_loss, 0
            torch.save({"state_dict": model.state_dict(), "img_size": args.img_size,
                        "seed": args.seed, "split_csv": str(args.split_csv),
                        "pretrained": not args.no_pretrained, "epoch": ep, "val_loss": v_loss}, ckpt)
            flag = "  <- saved (best val loss)"
        else:
            bad += 1
        print(f"ep {ep:02d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} "
              f"| val loss {v_loss:.4f} acc {v_acc:.3f} | {row['secs']}s{flag}")
        if bad >= args.patience:
            print(f"early stop (no val-loss improvement for {args.patience} epochs)")
            break

    pd.DataFrame(hist).to_csv(C.RESULTS_DIR / f"train_history_{tag}.csv", index=False)
    json.dump({"best_val_loss": best, "args": vars(args)}, open(C.RESULTS_DIR / f"train_info_{tag}.json", "w"), indent=2)
    print("best val loss", round(best, 4), "| checkpoint:", ckpt)


if __name__ == "__main__":
    main()
