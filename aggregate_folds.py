"""
aggregate_folds.py  -  pool the cross-validation folds into ONE result over ALL patients.

For every fold:   tau is chosen on that fold's VALIDATION patients only (same rule as escalate.py)
                  and applied once to that fold's TEST patients.
Then:             the test decisions of all folds are pooled (every patient appears once) and we report
                    * per-fold table with mean +- std
                    * pooled metrics for escalation, all-4, and each single magnification
                    * 95% patient-level bootstrap CIs (now over ~81 patients, not 13)
                    * how many benign / malignant patients the all-4 system gets mostly wrong

Run:  python aggregate_folds.py --preds results/preds_fold0.csv results/preds_fold1.csv ...
      (wildcards like results/preds_fold*.csv also work)
"""
import argparse
import glob

import numpy as np
import pandas as pd

import config as C
import escalate as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--prob-col", default="p_cal", choices=["p_cal", "p_raw"])
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--tag", default="kfold", help="prefix of the output files (use a new tag for a new run)")
    args = ap.parse_args()

    files = sorted({f for pat in args.preds for f in (glob.glob(pat) or [pat])})
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say(f"folds found: {len(files)}")

    rows, metas, ys, Ps, pe_all, ke_all, pa_all, ka_all = [], [], [], [], [], [], [], []
    for f in files:
        preds = pd.read_csv(f)
        Pv, mv = E.build_groups(preds[preds.split == "validation"], args.prob_col)
        Pt, mt = E.build_groups(preds[preds.split == "test"], args.prob_col)
        yv, yt = mv["label"].values, mt["label"].values
        sv, bv = E.sweep(Pv, yv), E.baselines(Pv, yv)
        all4_val = float(bv[bv.method == "all 4 (mean prob)"].balanced_acc.iloc[0])
        tau = E.choose_tau(sv, all4_val, args.margin)

        pe, ke, pp = E.run_policy(Pt, tau)
        pa, ka, pq = E.run_policy(Pt, 1.01)
        me, ma = E.metrics(yt, pe, pp, ke), E.metrics(yt, pa, pq, ka)
        rows.append({"file": f.replace("\\", "/").split("/")[-1], "test_patients": mt.patient_id.nunique(),
                     "tau": tau, "esc_acc": me["accuracy"], "all4_acc": ma["accuracy"],
                     "esc_bal": me["balanced_acc"], "all4_bal": ma["balanced_acc"],
                     "esc_spec": me["specificity"], "esc_mags": me["avg_mags"]})
        metas.append(mt); ys.append(yt); Ps.append(Pt)
        pe_all.append(pe); ke_all.append(ke); pa_all.append(pa); ka_all.append(ka)

    per = pd.DataFrame(rows)
    say("\nPER-FOLD (tau chosen on that fold's validation patients)")
    say(per.round(4).to_string(index=False))
    say("\nmean +- std over folds")
    for c in ["esc_acc", "all4_acc", "esc_bal", "all4_bal", "esc_spec", "esc_mags", "tau"]:
        say(f"  {c:9s} {per[c].mean():.4f} +- {per[c].std(ddof=1) if len(per) > 1 else float('nan'):.4f}")

    # ---- pooled over every patient (each tested exactly once) ------------------------
    meta = pd.concat(metas)
    y, P = np.concatenate(ys), np.vstack(Ps)
    pe, ke = np.concatenate(pe_all), np.concatenate(ke_all)
    pa, ka = np.concatenate(pa_all), np.concatenate(ka_all)
    per_fold_pats = [set(m.patient_id) for m in metas]
    for i in range(len(per_fold_pats)):
        for j in range(i + 1, len(per_fold_pats)):
            assert per_fold_pats[i].isdisjoint(per_fold_pats[j]), "a patient is in two test folds"

    say(f"\nPOOLED TEST: {len(y)} groups from {meta.patient_id.nunique()} patients "
        f"({meta.drop_duplicates('patient_id').label.eq(0).sum()} benign, "
        f"{meta.drop_duplicates('patient_id').label.eq(1).sum()} malignant)")
    pooled = [{"method": "escalation (per-fold tau)", **E.metrics(y, pe, pe * 0 + pe, ke)},
              {"method": "all 4 (mean prob)", **E.metrics(y, pa, pa * 0 + pa, ka)}]
    for i, m in enumerate(C.MAGS):
        pm = (P[:, i] >= 0.5).astype(int)
        pooled.append({"method": f"only {m}", **E.metrics(y, pm, P[:, i], np.ones(len(y)))})
    tab = pd.DataFrame(pooled).drop(columns=["auc", "n_groups"])
    say(tab.round(4).to_string(index=False))

    bs = E.bootstrap(meta, y, pe, ke, pa, ka)
    f2 = lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]"
    say("\n95% patient-level bootstrap CIs (pooled)")
    say(f"  escalation balanced acc : {f2(bs['esc_bal'])}")
    say(f"  all-4 balanced acc      : {f2(bs['all4_bal'])}")
    say(f"  difference (esc - all4) : {f2(bs['diff_bal'])}")
    say(f"  avg magnifications used : {f2(bs['esc_mags'])}")

    # ---- which patients does the all-4 system get mostly wrong? -------------------------
    d = pd.DataFrame({"patient_id": meta.patient_id.values, "label": y, "ok": (pa == y).astype(int)})
    pp_ = d.groupby("patient_id").agg(label=("label", "first"), acc=("ok", "mean"))
    say("\nPATIENT-LEVEL FAILURES of the all-4 system (patients with < 50% of cases correct)")
    for lab, name in ((0, "benign"), (1, "malignant")):
        s = pp_[pp_.label == lab]
        bad = s[s.acc < 0.5]
        say(f"  {name:9s}: {len(bad)} of {len(s)} patients  " + (", ".join(bad.index) if len(bad) else ""))

    per.to_csv(C.RESULTS_DIR / f"{args.tag}_per_fold.csv", index=False)
    tab.to_csv(C.RESULTS_DIR / f"{args.tag}_pooled.csv", index=False)
    pp_.to_csv(C.RESULTS_DIR / f"{args.tag}_patient_accuracy.csv")
    (C.RESULTS_DIR / f"{args.tag}_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    say(f"\nSaved {args.tag}_summary.txt / {args.tag}_per_fold.csv / {args.tag}_pooled.csv / {args.tag}_patient_accuracy.csv in results/")


if __name__ == "__main__":
    main()
