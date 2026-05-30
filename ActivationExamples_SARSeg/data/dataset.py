import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import lmdb
from safetensors.numpy import load
from ..config import replace_pixel_values_with_class_indices, pixel_value_to_class_index


class SARSegmentationDataset120(Dataset):
    def __init__(self, lmdb_path, matches, num_classes=20, transform=None):
        self.image_lmdb_file = lmdb_path
        self.env = None
        self.matches = matches
        self.num_classes = num_classes
        self.transform = transform
        self.open_env()

    def open_env(self):
        if self.env is None:
            print("Opening LMDB environment ...")
            self.env = lmdb.open(
                str(self.image_lmdb_file),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                map_size=8 * 1024**3,   # 8GB blocked for caching
                max_spare_txns=16,
            )

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        self.open_env()
        image_key, reference_key = self.matches[idx]

        # Retrieve data from LMDB
        with self.env.begin() as txn:
            # Load image data
            image_data = load(txn.get(image_key.encode()))
            image_bands = ["VH", "VV"]  # Sentinel 1 bands
            image_tensor = np.stack([image_data[band] for band in image_bands])

            # Load reference map
            mask_data = load(txn.get(reference_key.encode()))

        mask_data = mask_data["Data"]
        mask_indices = replace_pixel_values_with_class_indices(mask_data, pixel_value_to_class_index)

        # Ensure valid range
        mask_indices = np.clip(mask_indices, 1, self.num_classes-1)
        mask_indices_one_hot = F.one_hot(torch.tensor(mask_indices).long(), num_classes=self.num_classes).permute(2, 0, 1).float().numpy()  # (H, W, C) -> (C, H, W)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image_tensor, mask=mask_indices_one_hot)
            image_tensor = augmented["image"]
            mask_indices_one_hot = augmented["mask"]

        # Convert to tensors
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
        mask_indices_one_hot = torch.tensor(mask_indices_one_hot, dtype=torch.long)

        return image_tensor, mask_indices_one_hot
