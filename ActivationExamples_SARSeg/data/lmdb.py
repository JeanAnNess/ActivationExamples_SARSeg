import pandas as pd
import numpy as np
import lmdb
from safetensors.numpy import load


def match_keys(parquet_path):
    """
    Matches image keys with their corresponding reference map keys.
    Note: This function is specific to the BigEarthNet dataset. Please adjust as needed. As per their paper https://arxiv.org/pdf/2407.03653 p.3 they have a 2:1:1 train val test ratio which I will follow.

    Args:
        parquet_path (str): Path to the parquet file.

    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]: A tuple of (train_matches, validation_matches, test_matches) with a ratio of 2:1:1.

    """
    df = pd.read_parquet(parquet_path)
    df["reference_map"] = df["patch_id"].apply(lambda x: x + "_reference_map")
    df = df[["s1_name", "reference_map", "split"]]

    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    matches_train = []
    matches_validation = []
    matches_test = []

    for index, row in train.iterrows():
        matches_train.append((row["s1_name"], row["reference_map"]))

    for index, row in validation.iterrows():
        matches_validation.append((row["s1_name"], row["reference_map"]))

    for index, row in test.iterrows():
        matches_test.append((row["s1_name"], row["reference_map"]))

    return matches_train, matches_validation, matches_test


def get_image_and_mask_from_key(image_key, reference_key=None, lmdb_path=None):
    """
    Get the image and mask from the LMDB database based on the image key and reference key.

    Args:
        image_key (str): The image key.
        reference_key (str): The reference key.
        lmdb_path (str): Path to the LMDB database.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The image and mask.
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        image_data = load(txn.get(image_key.encode()))
        if reference_key is not None:
            mask_data = load(txn.get(reference_key.encode()))
            mask = mask_data["Data"]
        else:
            mask = None

    image = np.stack([image_data[band] for band in ["VH", "VV"]])
    return image, mask


def get_reference_ref(image_ref, parquet_path):
    """
    Returns the corresponding reference map key for a given image reference.
    """
    df = pd.read_parquet(parquet_path)
    df["reference_map"] = df["patch_id"] + "_reference_map"
    lookup = dict(zip(df["s1_name"], df["reference_map"]))
    return lookup[image_ref]
