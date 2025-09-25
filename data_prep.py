# data_prep.py  (robust version)
import random, shutil
from pathlib import Path

SRC_DIR = Path("datasets/leaves")
OUT_DIR = Path("leaf_dataset")
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SEED = 123

random.seed(SEED)

def clear_and_create(out_dir):
    # create root if missing
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in ("train","val","test"):
        p = out_dir / s
        if p.exists():
            if p.is_file():
                # if a file exists where a directory should be, remove it and create directory
                print(f"Removing file and creating directory: {p}")
                p.unlink()
                p.mkdir(parents=True, exist_ok=True)
            else:
                # it's already a directory — leave it (we won't delete its contents)
                print(f"Directory already exists (will reuse): {p}")
        else:
            p.mkdir(parents=True, exist_ok=True)

def main():
    if not SRC_DIR.exists():
        raise SystemExit(f"{SRC_DIR} not found")
    clear_and_create(OUT_DIR)
    for cls in sorted(SRC_DIR.iterdir()):
        if not cls.is_dir(): continue
        files = [p for p in cls.iterdir() if p.is_file()]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        for split_name, split_files in (("train",train_files),("val",val_files),("test",test_files)):
            dest = OUT_DIR / split_name / cls.name
            dest.mkdir(parents=True, exist_ok=True)
            for s in split_files:
                shutil.copy2(s, dest / s.name)
        print(f"{cls.name}: total {n}, train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")
    print("Created/updated leaf_dataset/ with train/val/test")

if __name__ == "__main__":
    main()
