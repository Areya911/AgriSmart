# rebuild_leaf_dataset.py
import random, shutil, sys
from pathlib import Path

SRC_DIR = Path("datasets/leaves")
OUT_DIR = Path("leaf_dataset")
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SEED = 12345
random.seed(SEED)

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def compute_splits(n):
    if n == 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    # n >= 3
    n_val = max(1, int(n * VAL_RATIO))
    n_test = max(1, int(n * TEST_RATIO))
    n_train = n - n_val - n_test
    # if rounding made train 0, adjust
    if n_train < 1:
        n_train = max(1, n - n_val - n_test)
        if n_train + n_val + n_test > n:
            n_val = max(0, n_val - 1)
    return n_train, n_val, n_test

def main():
    if not SRC_DIR.exists():
        print("Source folder missing:", SRC_DIR)
        sys.exit(1)

    # Remove old output
    if OUT_DIR.exists():
        print("Removing existing", OUT_DIR)
        try:
            shutil.rmtree(OUT_DIR)
        except Exception as e:
            print("Failed to remove existing leaf_dataset:", e)
            sys.exit(1)

    # Create splits
    ensure_dir(OUT_DIR)
    for split in ("train","val","test"):
        ensure_dir(OUT_DIR / split)

    classes = [d for d in sorted(SRC_DIR.iterdir()) if d.is_dir()]
    if not classes:
        print("No class folders found under", SRC_DIR)
        sys.exit(1)

    summary = []
    for cls in classes:
        files = [p for p in cls.iterdir() if p.is_file()]
        random.shuffle(files)
        n = len(files)
        n_train, n_val, n_test = compute_splits(n)
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        # copy
        for split_name, split_files in (("train",train_files), ("val",val_files), ("test",test_files)):
            dest_dir = OUT_DIR / split_name / cls.name
            ensure_dir(dest_dir)
            for f in split_files:
                try:
                    shutil.copy2(f, dest_dir / f.name)
                except Exception as e:
                    print("Failed to copy", f, "->", dest_dir, ":", e)
        summary.append((cls.name, n, len(train_files), len(val_files), len(test_files)))
        print(f"{cls.name}: total {n}, train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

    print("\nSummary complete. Now verifying output folders...\n")
    # verification: list counts per split
    for split in ("train","val","test"):
        print("=== ", split)
        split_dir = OUT_DIR / split
        for cls_name, total, t, v, te in summary:
            p = split_dir / cls_name
            cnt = 0
            if p.exists() and p.is_dir():
                cnt = len([x for x in p.iterdir() if x.is_file()])
            print(f"{cls_name} - {cnt}")
    print("\nRebuild finished. Good luck!")
    return

if __name__ == "__main__":
    main()

