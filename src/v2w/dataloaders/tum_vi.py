from torch.utils.data import DataLoader
from v2w.datasets.tum_vi.dataset import TUMVIDataset


def create_tumvi_dataloader(
    root,
    sequence,
    split="train",
    batch_size=4,
    num_workers=4,
    pin_memory=True,
):
    """
    Returns a PyTorch DataLoader for TUMVI dataset.

    Args:
        root (str or Path): Path to the 'data/tum_vi' folder
        sequence (str): Sequence folder name, e.g., 'dataset-corridor4_512_16'
        split (str): 'train', 'val', or 'test'
        batch_size (int)
        num_workers (int)
        pin_memory (bool)

    Returns:
        torch.utils.data.DataLoader
    """

    dataset = TUMVIDataset(
        root=root,
        sequence=sequence,
        split=split
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader