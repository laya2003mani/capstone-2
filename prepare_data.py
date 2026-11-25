"""
prepare_data.py  -  STEP 1: scan BreaKHis, build patient-level split, verify no leakage.

What it does
  1. Finds every image under DATASET_ROOT/<mag>/<class>/.
  2. Reads the PATIENT id and the image sequence number from the filename:
         SOB_B_A-14-22549AB-40-001.png  ->  patient "14-22549AB", mag 40, seq 001
  3. Splits PATIENTS (not images) into train / validation / test, separately
     inside benign and malignant so both classes appear in every split.
  4. Builds a GROUP id  =  patient + sequence number.  A group is "complete"
     when it has all four magnifications. Escalation is evaluated on complete groups.
  5. Asserts that no patient appears in two splits.

Run:  python prepare_data.py --seed 42                # one split (split_seed42.csv)
      python prepare_data.py --folds 5 --seed 42      # 5-fold patient-level cross-validation
         -> split_fold0.csv ... split_fold4.csv. In fold i the TEST patients are fold i,
            the VALIDATION patients are fold i+1, the rest train. Every patient is tested
            exactly once, so results can be pooled over all patients.
"""
import argparse
import random
import re

import pandas as pd

import config as C

# SOB_<B|M>_<subtype>-<year>-<slide>-<mag>-<seq>
NAME_RE = re.compile(
    r"SOB_[BM]_[A-Z]+-(?P<patient>\d+-\d+[A-Z]*)-(?P<mag>\d+)-(?P<seq>\d+)$",
    re.IGNORECASE,
)


def scan_dataset():
    rows = []
    for mag in C.MAGS:
        for label_name in C.CLASSES:
            folder = C.DATASET_ROOT / mag / label_name
            if not folder.exists():
                print(f"[warn] folder missing: {folder}")
                continue
            for p in sorted(folder.rglob("*")):
                if p.is_file() and p.suffix.lower() in C.IMAGE_EXTENSIONS:
                    m = NAME_RE.search(p.stem)
                    rows.append({
                        "image_path": str(p),
                        "filename": p.name,
                        "patient_id": m.group("patient").upper() if m else None,
                        "seq": int(m.group("seq")) if m else None,
                        "mag_in_name": m.group("mag") + "X" if m else None,
                        "magnification": mag,
                        "label_name": label_name,
                        "label": C.CLASSES.index(label_name),
                    })
    return pd.DataFrame(rows)


def split_patients(patients, rng, val_frac, test_frac):
    """Shuffle one class's patients and cut them; every split gets >= 1 patient."""
    patients = sorted(patients)
    rng.shuffle(patients)
    n = len(patients)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    test = patients[:n_test]
    val = patients[n_test:n_test + n_val]
    train = patients[n_test + n_val:]
    return train, val, test


def assign_folds(p2l, k, seed):
    """Class-stratified fold id (0..k-1) for every patient; one counter runs across classes
    so fold sizes stay balanced."""
    rng = random.Random(seed)
    fold_of, c = {}, 0
    for cls in (0, 1):
        pats = sorted(p2l[p2l == cls].index)
        rng.shuffle(pats)
        for p in pats:
            fold_of[p] = c % k
            c += 1
    return fold_of


def check_no_leak(df):
    ids = {s: set(df[df.split == s].patient_id) for s in ("train", "validation", "test")}
    assert ids["train"].isdisjoint(ids["validation"]), "LEAK train/val"
    assert ids["train"].isdisjoint(ids["test"]), "LEAK train/test"
    assert ids["validation"].isdisjoint(ids["test"]), "LEAK val/test"
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="split seed")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=0, help="K>1: write K cross-validation splits instead")
    args = ap.parse_args()

    print("=" * 65, "\nSCANNING", C.DATASET_ROOT, "\n" + "=" * 65)
    df = scan_dataset()
    if df.empty:
        raise SystemExit("No images found. Fix DATASET_ROOT in config.py")
    print(f"Images found: {len(df)}")

    # ---- filename parsing must succeed for every image -------------------
    bad = df[df["patient_id"].isna()]
    if len(bad):
        print(bad["filename"].head(20).to_string(index=False))
        raise SystemExit("Some filenames did not match the BreaKHis pattern. Fix before splitting.")

    mismatch = df[df["mag_in_name"] != df["magnification"]]
    if len(mismatch):
        print(f"[warn] {len(mismatch)} images sit in a folder that disagrees with the "
              f"magnification in their filename. Folder name is used.")

    # ---- one patient = one class -----------------------------------------
    n_cls = df.groupby("patient_id")["label"].nunique()
    if (n_cls > 1).any():
        raise SystemExit(f"Patients with both classes: {list(n_cls[n_cls > 1].index)}")

    p2l = df.groupby("patient_id")["label"].first()

    # ---- groups: same patient + same sequence number across magnifications
    df["group_id"] = df["patient_id"] + "_" + df["seq"].astype(str).str.zfill(3)
    mags_per_group = df.groupby("group_id")["magnification"].nunique()
    df["group_complete"] = df["group_id"].map(mags_per_group) == len(C.MAGS)

    if args.folds > 1:
        k = args.folds
        fold_of = assign_folds(p2l, k, args.seed)
        df["fold"] = df["patient_id"].map(fold_of)
        print(f"\nUnique patients: {df.patient_id.nunique()} | benign: {int((p2l == 0).sum())} "
              f"| malignant: {int((p2l == 1).sum())}   ->  {k}-fold patient-level cross-validation")
        for i in range(k):
            d = df.copy()
            d["split"] = "train"
            d.loc[d["fold"] == (i + 1) % k, "split"] = "validation"
            d.loc[d["fold"] == i, "split"] = "test"
            ids = check_no_leak(d)
            out = C.DATA_DIR / f"split_fold{i}.csv"
            d.to_csv(out, index=False)
            tp = d[d.split == "test"].drop_duplicates("patient_id")["label"].value_counts().to_dict()
            ng = d[(d.split == "test") & d.group_complete].group_id.nunique()
            print(f"fold {i}: patients train/val/test = {len(ids['train'])}/{len(ids['validation'])}/"
                  f"{len(ids['test'])} | test benign patients {tp.get(0, 0)}, malignant {tp.get(1, 0)} "
                  f"| complete test groups {ng}")
        print("\nPATIENT LEAKAGE: NONE in any fold (asserted). Each patient is in TEST exactly once.")
        print(f"Saved: {C.DATA_DIR}/split_fold0.csv ... split_fold{k - 1}.csv")
        return

    # ---- patient-level, class-stratified split (single split) ------------
    rng = random.Random(args.seed)
    split_of = {}
    for cls in (0, 1):
        pats = list(p2l[p2l == cls].index)
        tr, va, te = split_patients(pats, rng, args.val_frac, args.test_frac)
        split_of.update({p: "train" for p in tr})
        split_of.update({p: "validation" for p in va})
        split_of.update({p: "test" for p in te})
    df["split"] = df["patient_id"].map(split_of)
    ids = check_no_leak(df)

    out = C.DATA_DIR / f"split_seed{args.seed}.csv"
    df.to_csv(out, index=False)

    # ---- report ------------------------------------------------------------
    print("\nUnique patients:", df.patient_id.nunique(),
          "| benign:", int((p2l == 0).sum()), "| malignant:", int((p2l == 1).sum()))
    print("\nPatients per split:")
    print({s: len(v) for s, v in ids.items()})
    print("\nImages by split x class:")
    print(pd.crosstab(df["split"], df["label_name"]))
    print("\nImages by split x magnification:")
    print(pd.crosstab(df["split"], df["magnification"]))
    comp = df[df.group_complete].drop_duplicates("group_id")
    print("\nComplete 4-magnification groups per split (used by escalation):")
    print(comp["split"].value_counts().to_string())
    print("\nPATIENT LEAKAGE: NONE (asserted)")
    print("Saved:", out)


if __name__ == "__main__":
    main()
