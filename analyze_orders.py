"""
analyze_orders.py  -  OFFLINE analysis of the 5-fold predictions (no retraining, runs in seconds).

Questions it answers
  1. Does escalating from 40X first make sense?  It compares
       - escalation in the fixed order 40X > 100X > 200X > 400X
       - escalation in an order chosen on each fold's VALIDATION patients (best magnification first)
       - the single magnification that was best on validation (cost 1)
       - every fixed subset of magnifications (mean probability), e.g. 40X+100X
  2. Are the differences real?  Patient-level bootstrap CIs for the key comparisons.
  3. Which benign / malignant SUBTYPES contain the patients the system gets mostly wrong?

Important honesty rule: everything "chosen" here (tau, order, best single magnification) is chosen on
VALIDATION patients only. The pooled fixed-subset table is descriptive (it was NOT used for choosing).

Run:  python analyze_orders.py --preds "results/preds_fold*.csv"
"""
import argparse
import glob
import re
from itertools import combinations

import numpy as np
import pandas as pd

import config as C
import escalate as E

DEFAULT_ORDER = [0, 1, 2, 3]  # 40X, 100X, 200X, 400X


def subtype_of(fname):
    m = re.match(r"SOB_[BM]_([A-Z]+)-", str(fname), re.IGNORECASE)
    return m.group(1).upper() if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--prob-col", default="p_cal", choices=["p_cal", "p_raw"])
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. subw")
    args = ap.parse_args()
    files = sorted({f for pat in args.preds for f in (glob.glob(pat) or [pat])})
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    nm = lambda order: ">".join(C.MAGS[i] for i in order)

    res, ys, metas, subtype = {}, [], [], {}
    add = lambda name, pred, k: (res.setdefault(name, {"pred": [], "k": []})["pred"].append(pred),
                                 res[name]["k"].append(k))
    say(f"folds: {len(files)}")
    say("\nORDER CHOSEN ON VALIDATION IN EACH FOLD (best single magnification first)")
    for f in files:
        preds = pd.read_csv(f)
        for p, fn in zip(preds.patient_id, preds.filename):
            subtype[p] = subtype_of(fn)
        Pv, mv = E.build_groups(preds[preds.split == "validation"], args.prob_col)
        Pt, mt = E.build_groups(preds[preds.split == "test"], args.prob_col)
        yv, yt = mv["label"].values, mt["label"].values
        val_bal = [E.metrics(yv, (Pv[:, i] >= .5).astype(int), Pv[:, i], np.ones(len(yv)))["balanced_acc"]
                   for i in range(4)]
        order = [int(i) for i in np.argsort(-np.array(val_bal), kind="stable")]
        say(f"  {f.replace(chr(92), '/').split('/')[-1]}: {nm(order)}   (validation balanced acc per magnification: "
            + ", ".join(f"{C.MAGS[i]} {val_bal[i]:.3f}" for i in range(4)) + ")")

        def esc(o):
            Pvo, Pto = Pv[:, o], Pt[:, o]
            pa, ka, pp = E.run_policy(Pvo, 1.01)
            all4_val = E.metrics(yv, pa, pp, ka)["balanced_acc"]
            tau = E.choose_tau(E.sweep(Pvo, yv), all4_val, args.margin)
            pred, k, _ = E.run_policy(Pto, tau)
            return pred, k

        add("ESC fixed order 40X>100X>200X>400X", *esc(DEFAULT_ORDER))
        add("ESC order chosen on validation", *esc(order))
        add("single magnification chosen on validation", (Pt[:, order[0]] >= .5).astype(int), np.ones(len(yt)))
        for r in range(1, 5):
            for sub in combinations(range(4), r):
                add("fixed " + "+".join(C.MAGS[i] for i in sub),
                    (Pt[:, list(sub)].mean(1) >= .5).astype(int), np.full(len(yt), r))
        ys.append(yt); metas.append(mt)

    y, meta = np.concatenate(ys), pd.concat(metas)
    R = {n: (np.concatenate(v["pred"]), np.concatenate(v["k"])) for n, v in res.items()}
    say(f"\nPOOLED TEST: {len(y)} groups, {meta.patient_id.nunique()} patients")
    rows = [{"method": n, **{k: v for k, v in E.metrics(y, p, p, k_).items() if k not in ("auc", "n_groups", "f1")}}
            for n, (p, k_) in R.items()]
    tab = pd.DataFrame(rows).sort_values(["avg_mags", "balanced_acc"], ascending=[True, False])
    say(tab.round(4).to_string(index=False))
    say("(rows starting with 'fixed' are descriptive only; nothing was chosen from them)")

    f2 = lambda t: f"[{t[0]:+.3f}, {t[1]:+.3f}]"
    say("\nPAIRED COMPARISONS (balanced accuracy difference, 95% patient-level bootstrap CI)")
    A, S = "ESC order chosen on validation", "single magnification chosen on validation"
    D, F4 = "ESC fixed order 40X>100X>200X>400X", "fixed 40X+100X+200X+400X"
    for a, b in ((A, F4), (A, S), (D, F4), (D, S), (A, D)):
        bs = E.bootstrap(meta, y, R[a][0], R[a][1], R[b][0], R[b][1])
        ma, mb = E.metrics(y, *R[a][:1], R[a][0], R[a][1]), E.metrics(y, *R[b][:1], R[b][0], R[b][1])
        say(f"  {a}  minus  {b}\n      diff {ma['balanced_acc'] - mb['balanced_acc']:+.4f}  CI {f2(bs['diff_bal'])}"
            f"   | cost {ma['avg_mags']:.2f} vs {mb['avg_mags']:.2f}")

    # ---- subtype failures (all-4 system) ------------------------------------------------------
    p4, _ = R["fixed 40X+100X+200X+400X"]
    d = pd.DataFrame({"patient_id": meta.patient_id.values, "label": y, "ok": (p4 == y).astype(int)})
    pp = d.groupby("patient_id").agg(label=("label", "first"), acc=("ok", "mean"))
    pp["subtype"] = pp.index.map(subtype)
    pp["failed"] = pp.acc < 0.5
    say("\nSUBTYPES OF PATIENTS THE ALL-4 SYSTEM GETS MOSTLY WRONG (< 50% of cases correct)")
    for lab, name in ((0, "benign (A adenosis, F fibroadenoma, PT phyllodes, TA tubular adenoma)"),
                      (1, "malignant (DC ductal, LC lobular, MC mucinous, PC papillary)")):
        s = pp[pp.label == lab].groupby("subtype").agg(patients=("failed", "size"), failed=("failed", "sum"),
                                                        mean_acc=("acc", "mean")).round(3)
        say(f"\n{name}")
        say(s.to_string())
    say("\nfailed patients: " + ", ".join(f"{p} ({r.subtype})" for p, r in pp[pp.failed].iterrows()))

    sfx = f"_{args.tag}" if args.tag else ""
    tab.to_csv(C.RESULTS_DIR / f"order_analysis{sfx}.csv", index=False)
    pp.to_csv(C.RESULTS_DIR / f"subtype_failures{sfx}.csv")
    (C.RESULTS_DIR / f"order_analysis{sfx}.txt").write_text("\n".join(lines), encoding="utf-8")
    say(f"\nSaved order_analysis{sfx}.txt / .csv / subtype_failures{sfx}.csv in results/")


if __name__ == "__main__":
    main()
