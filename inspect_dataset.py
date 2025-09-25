# inspect_dataset.py
from pathlib import Path
p = Path("datasets/leaves")
if not p.exists():
    print("datasets/leaves missing")
else:
    for d in sorted(p.iterdir()):
        if d.is_dir():
            print(d.name, len(list(d.glob("*.*"))))
