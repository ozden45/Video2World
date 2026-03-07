import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
    cam0_dir = root / "raw" / sequence / "mav0" / "cam0" / "data"
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

    logger.info("Split saved to: %s", out_file)
    logger.info(
        "Dataset split sizes | train=%d val=%d test=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"])
    )

    