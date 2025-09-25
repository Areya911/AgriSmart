# remove_classes.py
import shutil, os, random
from pathlib import Path
import math

ROOT = Path("datasets/leaves")   # original per-class folders
ARCHIVE = ROOT / "archived_rare" # where we move removed classes (archive)
OUT = Path("leaf_dataset")       # output dataset root (train/val/test)
RANDOM_SEED = 42

# EDIT THIS LIST to the class folder names you want to archive (move)
TO_REMOVE = [
    "diseased_two-spotted_spider_mite",
    "diseased_leaf_Mold",
    "diseased_scab"
]

# train/val/test ratios
RATIO = (0.8, 0.1, 0.1)

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def archive_classes():
    ensure(ARCHIVE)
    for cls in TO_REMOVE:
        src = ROOT / cls
        if src.exists():
            dst = ARCHIVE / cls
            print(f"Archiving {src} -> {dst}")
            if dst.exists():
                print("  destination already exists, skipping move.")
            else:
                shutil.move(str(src), str(dst))
        else:
            print(f"Class folder not found (skipping): {src}")

def rebuild_splits():
    # remove old output if exists
    if OUT.exists():
        print("Removing existing", OUT)
        shutil.rmtree(OUT)
    ensure(OUT)
    train_dir = OUT / "train"
    val_dir = OUT / "val"
    test_dir = OUT / "test"
    ensure(train_dir); ensure(val_dir); ensure(test_dir)

    classes = sorted([p.name for p in ROOT.iterdir() if p.is_dir()])
    print("Classes to include:", classes)

    random.seed(RANDOM_SEED)
    for cls in classes:
        src = ROOT / cls
        files = [p for p in src.iterdir() if p.is_file()]
        random.shuffle(files)
        n = len(files)
        n_train = max(1, math.floor(n * RATIO[0]))
        n_val = max(1, math.floor(n * RATIO[1]))
        n_test = n - n_train - n_val
        # if rounding made test 0, adjust
        if n_test <= 0 and n_train > 1:
            n_train -= 1
            n_test = 1
        print(f"{cls}: total {n}, train {n_train}, val {n_val}, test {n_test}")

        # make folders
        (train_dir/cls).mkdir(parents=True, exist_ok=True)
        (val_dir/cls).mkdir(parents=True, exist_ok=True)
        (test_dir/cls).mkdir(parents=True, exist_ok=True)

        for p in files[:n_train]:
            shutil.copy2(str(p), str(train_dir/cls/p.name))
        for p in files[n_train:n_train+n_val]:
            shutil.copy2(str(p), str(val_dir/cls/p.name))
        for p in files[n_train+n_val:]:
            shutil.copy2(str(p), str(test_dir/cls/p.name))

    print("Rebuild complete. Output at:", OUT)

if __name__ == "__main__":
    archive_classes()
    rebuild_splits()
