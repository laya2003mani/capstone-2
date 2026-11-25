"""
patient_report.py  -  per-patient diagnostic.

Shows, for every validation/test patient, how many of their images the model gets right
at each magnification. If a patient is wrong at ALL magnifications, the error is a
PATIENT-level problem (stain, tissue type, label noise) and escalation cannot fix it.

Run:  python patient_report.py --preds results/preds_seed0.csv
"""
import argparse

import pandas as pd

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--prob-col", default="p_cal", choices=["p_cal", "p_raw"])
    args = ap.parse_args()
    tag = args.preds.replace("\\", "/").split("/")[-1].replace("preds_", "").replace(".csv", "")

    d = pd.read_csv(args.preds)
    d["correct"] = ((d[args.prob_col] >= 0.5).astype(int) == d["label"]).astype(int)
    rows = []
    for (split, pat), g in d.groupby(["split", "patient_id"]):
        r = {"split": split, "patient_id": pat, "class": C.CLASSES[int(g["label"].iloc[0])],
             "n_images": len(g), "acc_all_mags": round(g["correct"].mean(), 3)}
        for m in C.MAGS:
            gm = g[g.magnification == m]
            r[f"acc_{m}"] = round(gm["correct"].mean(), 3) if len(gm) else float("nan")
        r["mean_p_malignant"] = round(g[args.prob_col].mean(), 3)
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(["split", "class", "acc_all_mags"])
    out.to_csv(C.RESULTS_DIR / f"patient_report_{tag}.csv", index=False)

    pd.set_option("display.width", 200)
    for split in ("test", "validation"):
        s = out[out.split == split]
        print(f"\n=== {split.upper()} patients ({len(s)}) ===")
        print(s.drop(columns="split").to_string(index=False))
        worst = s[s.acc_all_mags < 0.5]
        print(f"\npatients with < 50% of images correct: {len(worst)} "
              f"({', '.join(worst.patient_id) if len(worst) else 'none'})")
    print("\nSaved:", C.RESULTS_DIR / f"patient_report_{tag}.csv")


if __name__ == "__main__":
    main()
