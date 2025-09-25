# count_classes.py
from pathlib import Path
root = Path("leaf_dataset")
for split in ("train","val","test"):
    print("=== ", split)
    for cls in sorted((root/split).iterdir()):
        if cls.is_dir():
            print(cls.name, len(list(cls.glob("*.*"))))
