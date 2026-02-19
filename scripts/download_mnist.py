#!/usr/bin/env python3
"""
Download MNIST and/or Fashion-MNIST and save as CSV for C++ loading.
Each row: 784 pixel values (0-255) + 1 label (0-9). Total 785 columns.
Output: data/mnist_train.csv, data/mnist_test.csv, data/fashion_mnist_*.csv

Requires: pip install torch torchvision
"""
from pathlib import Path
import sys
import csv

def try_torchvision():
    try:
        from torchvision import datasets
        return True, datasets
    except ImportError:
        return False, None

def save_dataset(dataset, path, max_rows=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    if max_rows is not None and max_rows < n:
        n = max_rows
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for i in range(n):
            img, label = dataset[i]
            if hasattr(img, "numpy"):
                row = img.numpy().flatten().tolist()
            else:
                import numpy as np
                row = np.array(img).flatten().tolist()
            row.append(int(label))
            w.writerow(row)
            if (i + 1) % 10000 == 0:
                print(f"  Written {i + 1}/{n} rows")
    print(f"  Saved {path} ({n} rows, 785 cols)")

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ok, datasets_mod = try_torchvision()
    if not ok:
        print("Install PyTorch and torchvision: pip install torch torchvision", file=sys.stderr)
        print("Then run: python scripts/download_mnist.py", file=sys.stderr)
        sys.exit(1)

    max_train = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--max-train" and i + 1 < len(argv):
            max_train = int(argv[i + 1])
            break
    which = [a for a in argv if a in ("mnist", "fashion")] or ["mnist", "fashion"]

    if "mnist" in which:
        print("Downloading MNIST...")
        train = datasets_mod.MNIST(root=str(root), train=True, download=True)
        test = datasets_mod.MNIST(root=str(root), train=False, download=True)
        print("Writing MNIST CSV...")
        save_dataset(train, data_dir / "mnist_train.csv", max_train)
        save_dataset(test, data_dir / "mnist_test.csv")
    if "fashion" in which:
        print("Downloading Fashion-MNIST...")
        train = datasets_mod.FashionMNIST(root=str(root), train=True, download=True)
        test = datasets_mod.FashionMNIST(root=str(root), train=False, download=True)
        print("Writing Fashion-MNIST CSV...")
        save_dataset(train, data_dir / "fashion_mnist_train.csv", max_train)
        save_dataset(test, data_dir / "fashion_mnist_test.csv")

    print("Done. Train with: build/train_mnist [--fashion] [--epochs N]")

if __name__ == "__main__":
    main()
