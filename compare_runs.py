"""
compare_runs.py  -  is run B better than run A?  (same 5 folds, same patients, paired comparison)

Typical use:  A = baseline (class-weighted loss)      results/preds_fold*.csv
              B = subtype-weighted loss               results/preds_subw_fold*.csv

For both runs it applies exactly the same rules (tau chosen on each fold's VALIDATION patients), pools the
test decisions of all folds, and reports
  * pooled metrics for: all 4 magnifications, escalation (40X>100X>200X>400X), each single magnification
  * PAIRED patient-level bootstrap CI of the balanced-accuracy difference (B minus A)
  * patient failures per tumour subtype, side by side (a patient "fails" if < 50% of its cases are correct
    with all 4 magnifications), and which patients changed from wrong to right or right to wrong

Run:  python compare_runs.py --a "results/preds_fold*.csv" --b "results/preds_subw_fold*.csv" --name-b subw
"""
import argparse
import glob
import re

import numpy as np
import pandas as pd

import config as C
import escalate as E


def subtype_of(fname):
    m = re.match(r"SOB_[BM]_([A-Z]+)-", str(fname), re.IGNORECASE)
    return m.group(1).upper() if m else "?"


def expand(patterns):
    return sorted({f for p in patterns for f in (glob.glob(p) or [p])})


def collect(files, prob_col, margin):
    """Pool the test decisions of all folds for one run."""
    out = {"y": [], "meta": [], "all4": [], "esc": [], "kesc": [], "P": []}
    subtype = {}
    for f in files:
        d = pd.read_csv(f)
        for p, fn in zip(d.patient_id, d.filename):
            subtype[p] = subtype_of(fn)
        Pv, mv = E.build_groups(d[d.split == "validation"], prob_col)
        Pt, mt = E.build_groups(d[d.split == "test"], prob_col)
        yv, yt = mv["label"].values, mt["label"].values
        pa, ka, pp = E.run_policy(Pv, 1.01)
        tau = E.choose_tau(E.sweep(Pv, yv), E.metrics(yv, pa, pp, ka)["balanced_acc"], margin)
        pe, ke, _ = E.run_policy(Pt, tau)
        p4, _, _ = E.run_policy(Pt, 1.01)
        out["y"].append(yt); out["meta"].append(mt); out["all4"].append(p4)
        out["esc"].append(pe); out["kesc"].append(ke); out["P"].append(Pt)
    r = {k: (pd.concat(v) if k == "meta" else (np.vstack(v) if k == "P" else np.concatenate(v))) for k, v in out.items()}
    r["subtype"] = subtype
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--name-a", default="baseline")
    ap.add_argument("--name-b", default="subw")
    ap.add_argument("--prob-col", default="p_cal", choices=["p_cal", "p_raw"])
    ap.add_argument("--margin", type=float, default=0.01)
    args = ap.parse_args()
    fa, fb = expand(args.a), expand(args.b)
    assert len(fa) == len(fb) and len(fa) > 0, f"need the same number of folds: {len(fa)} vs {len(fb)}"
    A, B = collect(fa, args.prob_col, args.margin), collect(fb, args.prob_col, args.margin)
    assert list(A["meta"].index) == list(B["meta"].index), "the two runs do not cover the same groups/folds"
    y, meta = A["y"], A["meta"]
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say(f"A = {args.name_a} ({len(fa)} folds)   B = {args.name_b} ({len(fb)} folds)")
    say(f"POOLED TEST: {len(y)} groups, {meta.patient_id.nunique()} patients\n")

    rows = []
    for nm, R in ((args.name_a, A), (args.name_b, B)):
        k4 = np.full(len(y), 4)
        rows.append({"run": nm, "method": "all 4 magnifications", **E.metrics(y, R["all4"], R["all4"], k4)})
        rows.append({"run": nm, "method": "escalation 40X>100X>200X>400X", **E.metrics(y, R["esc"], R["esc"], R["kesc"])})
        for i, m in enumerate(C.MAGS):
            pm = (R["P"][:, i] >= .5).astype(int)
            rows.append({"run": nm, "method": f"only {m}", **E.metrics(y, pm, pm, np.ones(len(y)))})
    tab = pd.DataFrame(rows).drop(columns=["auc", "n_groups", "f1"])
    say(tab.round(4).to_string(index=False))

    f2 = lambda t: f"[{t[0]:+.3f}, {t[1]:+.3f}]"
    say(f"\nPAIRED DIFFERENCE IN BALANCED ACCURACY ({args.name_b} minus {args.name_a}), 95% patient-level bootstrap CI")
    k4 = np.full(len(y), 4)
    for label, pb, kb, pa_, ka_ in (("all 4 magnifications", B["all4"], k4, A["all4"], k4),
                                    ("escalation 40X first", B["esc"], B["kesc"], A["esc"], A["kesc"])):
        bs = E.bootstrap(meta, y, pb, kb, pa_, ka_)
        d = E.metrics(y, pb, pb, kb)["balanced_acc"] - E.metrics(y, pa_, pa_, ka_)["balanced_acc"]
        say(f"  {label:24s} diff {d:+.4f}  CI {f2(bs['diff_bal'])}   | avg magnifications {kb.mean():.2f} vs {ka_.mean():.2f}")

    # ---- per-subtype patient failures, side by side -------------------------------------
    def per_patient(R):
        d = pd.DataFrame({"patient_id": meta.patient_id.values, "label": y, "ok": (R["all4"] == y).astype(int)})
        return d.groupby("patient_id").agg(label=("label", "first"), acc=("ok", "mean"))
    pa_, pb_ = per_patient(A), per_patient(B)
    t = pa_.join(pb_[["acc"]], rsuffix="_b").rename(columns={"acc": "acc_a", "acc_b": "acc_b"})
    t["subtype"] = t.index.map(A["subtype"])
    t["fail_a"], t["fail_b"] = t.acc_a < .5, t.acc_b < .5
    s = t.groupby(["label", "subtype"]).agg(patients=("fail_a", "size"), failed_a=("fail_a", "sum"),
                                              failed_b=("fail_b", "sum"), mean_acc_a=("acc_a", "mean"),
                                              mean_acc_b=("acc_b", "mean")).round(3)
    s = s.rename(columns={"failed_a": f"failed_{args.name_a}", "failed_b": f"failed_{args.name_b}",
                          "mean_acc_a": f"acc_{args.name_a}", "mean_acc_b": f"acc_{args.name_b}"})
    say("\nPATIENT FAILURES BY SUBTYPE, all 4 magnifications  (label 0 = benign, 1 = malignant)")
    say(s.to_string())
    say(f"\ntotal failed patients: {args.name_a} {int(t.fail_a.sum())}  ->  {args.name_b} {int(t.fail_b.sum())}  (of {len(t)})")
    for lab, nm in ((0, "benign"), (1, "malignant")):
        u = t[t.label == lab]
        say(f"  {nm:9s}: {int(u.fail_a.sum())} -> {int(u.fail_b.sum())} of {len(u)}")
    fixed = t[t.fail_a & ~t.fail_b]; broke = t[~t.fail_a & t.fail_b]
    say("fixed by B : " + (", ".join(f"{p} ({r.subtype})" for p, r in fixed.iterrows()) or "none"))
    say("broken by B: " + (", ".join(f"{p} ({r.subtype})" for p, r in broke.iterrows()) or "none"))

    tag = f"compare_{args.name_a}_vs_{args.name_b}"
    tab.to_csv(C.RESULTS_DIR / f"{tag}.csv", index=False)
    t.to_csv(C.RESULTS_DIR / f"{tag}_patients.csv")
    (C.RESULTS_DIR / f"{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
    say(f"\nSaved {tag}.txt / .csv / _patients.csv in results/")


if __name__ == "__main__":
    main()
