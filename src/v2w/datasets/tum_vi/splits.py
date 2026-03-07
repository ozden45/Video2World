import json
from pathlib import Path


def create_splits(
    root,
    sequence,
    train_ratio=0.8,
    val_ratio=0.1
):
    """
    Create temporal splits for a TUM-VI sequence.
    """

    root = Path(root)

    cam0_dir = (
        root
        / "raw"
        / sequence
        / "mav0"
        / "cam0"
        / "data"
    )

    files = sorted(cam0_dir.glob("*.png"))

    timestamps = [f.stem for f in files]

    n = len(timestamps)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": timestamps[:train_end],
        "val": timestamps[train_end:val_end],
        "test": timestamps[val_end:]
    }

    out_dir = root / "processed" / sequence / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "split.json"

    with open(out_file, "w") as f:
        json.dump(splits, f, indent=2)

    print("Split saved to:", out_file)
    print("Train:", len(splits["train"]))
    print("Val:", len(splits["val"]))
    print("Test:", len(splits["test"]))


if __name__ == "__main__":

    create_splits(
        root="data/tum_vi",
        sequence="dataset-corridor4_512_16"
    )
    
    