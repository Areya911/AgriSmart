# data_prep_min1.py - ensures at least 1 sample in val and test when possible
import random, shutil
from pathlib import Path

SRC_DIR = Path("datasets/leaves")
OUT_DIR = Path("leaf_dataset")
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SEED = 123
random.seed(SEED)

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def main():
    if not SRC_DIR.exists():
        raise SystemExit(f"{SRC_DIR} not found")
    # remove old split if present to ensure a fresh start
    if OUT_DIR.exists():
        try:
            shutil.rmtree(OUT_DIR)
        except Exception as e:
            print("Warning: could not remove existing", OUT_DIR, ":", e)
    ensure_dir(OUT_DIR)
    for cls in sorted(SRC_DIR.iterdir()):
        if not cls.is_dir(): continue
        files = [p for p in cls.iterdir() if p.is_file()]
        random.shuffle(files)
        n = len(files)
        if n == 0:
            print("Skipping empty class", cls.name)
            continue
        # allocate at least 1 for val/test if possible
        if n >= 3:
            n_val = max(1, int(n * VAL_RATIO))
            n_test = max(1, int(n * TEST_RATIO))
            n_train = n - n_val - n_test
            # adjust if rounding made train 0
            if n_train < 1:
                n_train = max(1, n - n_val - n_test)
                if n_train + n_val + n_test > n:
                    n_val = max(0, n_val - 1)
        else:
            if n == 1:
                n_train, n_val, n_test = 1, 0, 0
            else:
                n_train, n_val, n_test = n-1, 1, 0

        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]

        for split_name, split_files in (("train",train_files), ("val",val_files), ("test",test_files)):
            dest = OUT_DIR / split_name / cls.name
            ensure_dir(dest)
            for s in split_files:
                shutil.copy2(s, dest / s.name)
        print(f"{cls.name}: total {n}, train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

    print("Created leaf_dataset/ with stratified splits (min 1 where possible)")

if __name__ == "__main__":
    main()
