"""
escalate.py  -  STEP 4: the confidence-gated escalation policy (Objective 1 + 6).

POLICY (for one case = one matched group of 40X/100X/200X/400X images)
    k = 1
    loop:
        evidence = average of malignant-probabilities of the first k magnifications
        confidence = max(evidence, 1 - evidence)
        if confidence >= tau  or  k == 4:  STOP, predict malignant if evidence >= 0.5
        else: k = k + 1          (acquire the next magnification)

HOW tau IS CHOSEN (no test-set peeking)
    * sweep tau on VALIDATION groups
    * pick the CHEAPEST tau whose validation balanced accuracy is within --margin of the
      "always use all four magnifications" baseline
    * apply that one tau to the TEST groups and report it with patient-level bootstrap CIs
    * the full test curve is also saved, but only as a description, never for choosing tau

Run:  python escalate.py --preds results/preds_seed0.csv
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import config as C

TAUS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 1.01]  # 1.01 = never stop early


# ------------------------------------------------------------------ data
def build_groups(preds, prob_col):
    d = preds[preds.group_complete].drop_duplicates(["group_id", "magnification"])
    P = d.pivot(index="group_id", columns="magnification", values=prob_col)[C.MAGS].dropna()
    meta = d.drop_duplicates("group_id").set_index("group_id").loc[P.index, ["patient_id", "label", "split"]]
    return P.values, meta


# ------------------------------------------------------------------ policy
def run_policy(P, tau):
    n = len(P)
    cum = np.cumsum(P, axis=1) / np.arange(1, 5)       # running mean after k magnifications
    conf = np.maximum(cum, 1 - cum)
    stop = conf >= tau
    stop[:, -1] = True                                  # after 400X we must stop
    k = stop.argmax(1)                                  # first True
    p_final = cum[np.arange(n), k]
    return (p_final >= 0.5).astype(int), k + 1, p_final


def metrics(y, pred, p, cost):
    tp = ((y == 1) & (pred == 1)).sum(); fn = ((y == 1) & (pred == 0)).sum()
    tn = ((y == 0) & (pred == 0)).sum(); fp = ((y == 0) & (pred == 1)).sum()
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {
        "accuracy": (tp + tn) / len(y), "balanced_acc": (sens + spec) / 2,
        "sensitivity": sens, "specificity": spec,
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
        "avg_mags": float(np.mean(cost)), "n_groups": len(y),
    }


def sweep(P, y):
    rows = []
    for tau in TAUS:
        pred, k, p = run_policy(P, tau)
        rows.append({"tau": tau, **metrics(y, pred, p, k)})
    return pd.DataFrame(rows)


def baselines(P, y):
    rows = []
    for i, m in enumerate(C.MAGS):
        rows.append({"method": f"only {m}", **metrics(y, (P[:, i] >= 0.5).astype(int), P[:, i], np.ones(len(y)))})
    pred, k, p = run_policy(P, 1.01)
    rows.append({"method": "all 4 (mean prob)", **metrics(y, pred, p, k)})
    return pd.DataFrame(rows)


def choose_tau(val_sweep, val_all4_bal, margin):
    ok = val_sweep[val_sweep.balanced_acc >= val_all4_bal - margin]
    ok = ok.sort_values(["avg_mags", "balanced_acc"], ascending=[True, False])
    return float(ok.iloc[0]["tau"])


# ------------------------------------------------------------------ bootstrap over PATIENTS
def patient_counts(meta, y, pred, k):
    d = pd.DataFrame({
        "pat": meta["patient_id"].values,
        "tp": ((y == 1) & (pred == 1)).astype(int), "fn": ((y == 1) & (pred == 0)).astype(int),
        "tn": ((y == 0) & (pred == 0)).astype(int), "fp": ((y == 0) & (pred == 1)).astype(int),
        "cost": k, "n": 1,
    })
    return d.groupby("pat").sum().loc[sorted(d.pat.unique())].values.astype(float)


def boot_stats(c):  # c columns: tp fn tn fp cost n
    with np.errstate(invalid="ignore", divide="ignore"):
        bal = 0.5 * (c[:, 0] / (c[:, 0] + c[:, 1]) + c[:, 2] / (c[:, 2] + c[:, 3]))
    return bal, c[:, 4] / c[:, 5]


def bootstrap(meta, y, pred_a, k_a, pred_b, k_b, B=2000, seed=0):
    ca, cb = patient_counts(meta, y, pred_a, k_a), patient_counts(meta, y, pred_b, k_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ca), size=(B, len(ca)))
    ba, ma = boot_stats(ca[idx].sum(1)); bb, _ = boot_stats(cb[idx].sum(1))
    ci = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5]))
    return {"esc_bal": ci(ba), "all4_bal": ci(bb), "diff_bal": ci(ba - bb), "esc_mags": ci(ma)}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--prob-col", default="p_cal", choices=["p_cal", "p_raw"])
    ap.add_argument("--margin", type=float, default=0.01,
                    help="allowed drop in validation balanced accuracy vs all-4 baseline")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.preds.replace("\\", "/").split("/")[-1].replace("preds_", "").replace(".csv", "")

    preds = pd.read_csv(args.preds)
    Pv, mv = build_groups(preds[preds.split == "validation"], args.prob_col)
    Pt, mt = build_groups(preds[preds.split == "test"], args.prob_col)
    yv, yt = mv["label"].values, mt["label"].values
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    say(f"validation: {len(yv)} complete groups, {mv.patient_id.nunique()} patients")
    say(f"test      : {len(yt)} complete groups, {mt.patient_id.nunique()} patients  "
        f"<- results depend on very few patients, read CIs, not just point estimates")

    sv, st = sweep(Pv, yv), sweep(Pt, yt)
    bv, bt = baselines(Pv, yv), baselines(Pt, yt)
    all4_val_bal = float(bv[bv.method == "all 4 (mean prob)"].balanced_acc.iloc[0])

    tau = choose_tau(sv, all4_val_bal, args.margin)
    say(f"\nchosen tau (from VALIDATION only, margin {args.margin}): {tau}")

    # ---- TEST result at the chosen tau --------------------------------------
    pred_e, k_e, p_e = run_policy(Pt, tau)
    pred_a, k_a, p_a = run_policy(Pt, 1.01)
    m_e = metrics(yt, pred_e, p_e, k_e)
    say("\nTEST RESULT at chosen tau")
    say(pd.DataFrame([m_e]).round(4).to_string(index=False))
    say("\nstop level (how many magnifications were used):")
    say(pd.Series(k_e).value_counts(normalize=True).sort_index().round(3).rename(lambda i: f"{i} mag(s)").to_string())

    bs = bootstrap(mt, yt, pred_e, k_e, pred_a, k_a)
    f = lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]"
    say("\n95% patient-level bootstrap CIs (test):")
    say(f"  escalation balanced acc : {f(bs['esc_bal'])}")
    say(f"  all-4 balanced acc      : {f(bs['all4_bal'])}")
    say(f"  difference (esc - all4) : {f(bs['diff_bal'])}   <- if this contains 0, no detectable accuracy loss")
    say(f"  avg magnifications used : {f(bs['esc_mags'])}")

    say("\nBASELINES on TEST")
    say(bt.round(4).to_string(index=False))
    say("\nFULL TEST SWEEP (descriptive only, NOT used to choose tau)")
    say(st.round(4).to_string(index=False))

    # ---- save ---------------------------------------------------------------
    sv.to_csv(C.RESULTS_DIR / f"escalation_sweep_val_{tag}.csv", index=False)
    st.to_csv(C.RESULTS_DIR / f"escalation_sweep_test_{tag}.csv", index=False)
    bt.to_csv(C.RESULTS_DIR / f"baselines_test_{tag}.csv", index=False)
    (C.RESULTS_DIR / f"escalation_report_{tag}.txt").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, sw, bl, name in ((axes[0], sv, bv, "Validation"), (axes[1], st, bt, "Test")):
        ax.plot(sw.avg_mags, sw.balanced_acc, "o-", label="escalation (tau sweep)")
        for _, r in bl.iterrows():
            ax.scatter(r.avg_mags, r.balanced_acc, marker="s" if "all" in r.method else "^", s=55, label=r.method)
        row = sw[sw.tau == tau].iloc[0]
        ax.scatter(row.avg_mags, row.balanced_acc, s=220, facecolors="none", edgecolors="k", linewidths=2, label=f"chosen tau={tau}")
        ax.set_title(name); ax.set_xlabel("average magnifications used per case"); ax.grid(alpha=.3)
    axes[0].set_ylabel("balanced accuracy"); axes[1].legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / f"escalation_curve_{tag}.png", dpi=150)
    say("\nSaved report, csv files and plot in results/")


if __name__ == "__main__":
    main()
