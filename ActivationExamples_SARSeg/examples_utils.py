"""
Utilities for Activation-Based Example Extraction and Visualization
Authored by Janes Sanne
"""

"""
Imports
"""
# Standard Libraries
import random
import ast

# Plotting
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# Machine Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss, JaccardLoss, FocalLoss
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from scipy.spatial.distance import cdist

# Data Handling
import pandas as pd
import numpy as np
import lmdb
from safetensors.numpy import load
import heapq

# Progress Bar
from tqdm import tqdm

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

""" 
General Parameters
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

color_map = {
    1:  [206, 206, 206],  # Urban fabric (Gray)
    2:  [64,  64,  64],   # Industrial or commercial units (Light Gray)
    3:  [254, 215, 119],  # Arable land (Orange)
    4:  [254, 187, 71],   # Permanent crops (Yellow)
    5:  [253, 152, 40],   # Pastures (Green-Yellow)
    6:  [239, 120, 24],   # Complex cultivation patterns (Forest Green)
    7:  [216, 89,  8],    # Land principally occupied by agriculture, with significant areas of natural vegetation (light green-yellow)
    8:  [183, 66,  2],    # Agro-forestry areas (Olive)
    9:  [200, 232, 154],  # Broad-leaved forest (Dark Green)
    10: [162, 216, 137],  # Coniferous forest (Green)
    11: [119, 197, 120],  # Mixed forest (Forest Green)
    12: [75,  176, 98],   # Natural grassland and sparsely vegetated areas (Yellow-Green)
    13: [46,  146, 76],   # Moors, heathland and sclerophyllous vegetation (Olive Drab)
    14: [21,  120, 62],   # Transitional woodland, shrub (Light Green)
    15: [210, 180, 140],  # Beaches, dunes, sands (Tan)
    16: [182, 212, 233],  # Inland wetlands (Cyan)
    17: [120, 181, 216],  # Coastal wetlands (Deep Sky Blue)
    18: [63,  143, 196],  # Inland waters (Blue)
    19: [23,  100, 171],  # Marine waters (Midnight Blue)
    20: [255, 255, 255],  # Unlabeled (White)
}

groups = {
    "urban":               [1, 2],
    "agriculture":         [3, 4, 5, 6, 7, 8],
    "forest":              [9, 10, 11, 12, 13, 14],
    "sand":                [15],
    "water_wetlands":      [16, 17, 18, 19],
    "unlabeled":           [20],
}

proximity_lists = list(groups.values())

class_name_to_index = {
    "Urban fabric": 1,
    "Industrial or commercial units": 2,
    "Arable land": 3,
    "Permanent crops": 4,
    "Pastures": 5,
    "Complex cultivation patterns": 6,
    "Land principally occupied by agriculture, with significant areas of natural vegetation" : 7,
    "Agro-forestry areas": 8,
    "Broad-leaved forest": 9,
    "Coniferous forest": 10,
    "Mixed forest": 11,
    "Natural grassland and sparsely vegetated areas": 12,
    "Moors, heathland and sclerophyllous vegetation": 13,
    "Transitional woodland, shrub": 14,
    "Beaches, dunes, sands": 15,
    "Inland wetlands": 16,
    "Coastal wetlands": 17,
    "Inland waters": 18,
    "Marine waters": 19,
    "Unlabeled": 20,
}

# According to https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf match pixel values to classes
pixel_value_to_class_name = {
    111: "Urban fabric",
    112: "Urban fabric",
    121: "Industrial or commercial units",
    122: "Unlabeled",
    123: "Unlabeled",
    124: "Unlabeled",
    131: "Unlabeled",
    132: "Unlabeled",
    133: "Unlabeled",
    141: "Unlabeled",
    142: "Unlabeled",
    211: "Arable land",
    212: "Arable land",
    213: "Arable land",
    221: "Permanent crops",
    222: "Permanent crops",
    223: "Permanent crops",
    231: "Pastures",
    241: "Permanent crops",
    242: "Complex cultivation patterns",
    243: "Land principally occupied by agriculture, with significant areas of natural vegetation",
    244: "Agro-forestry areas",
    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    321: "Natural grassland and sparsely vegetated areas",
    322: "Moors, heathland and sclerophyllous vegetation",
    323: "Moors, heathland and sclerophyllous vegetation",
    324: "Transitional woodland, shrub",
    331: "Beaches, dunes, sands",
    332: "Unlabeled",
    333: "Natural grassland and sparsely vegetated areas",
    334: "Unlabeled",
    335: "Unlabeled",
    411: "Inland wetlands",
    412: "Inland wetlands",
    421: "Coastal wetlands",
    422: "Coastal wetlands",
    423: "Unlabeled",
    511: "Inland waters",
    512: "Inland waters",
    521: "Marine waters",
    522: "Marine waters",
    523: "Marine waters",
    999: "Unlabeled"
}

pixel_value_to_class_index = {111: 1, 112: 1, 121: 2, 122: 20, 123: 20, 124: 20, 131: 20, 132: 20, 133: 20, 141: 20, 142: 20, 211: 3, 212: 3, 213: 3, 221: 4, 222: 4, 223: 4, 231: 5, 241: 4, 242: 6, 243: 7, 244: 8, 311: 9, 312: 10, 313: 11, 321: 12, 322: 13, 323: 13, 324: 14, 331: 15, 332: 20, 333: 12, 334: 20, 335: 20, 411: 16, 412: 16, 421: 17, 422: 17, 423: 20, 511: 18, 512: 18, 521: 19, 522: 19, 523: 19, 999: 20}

LAYER_SHAPES = {
    "encoder.layer1": (1, 256, 30, 30),
    "encoder.layer2": (1, 512, 15, 15),
    "encoder.layer3": (1, 1024, 8, 8),
    "encoder.layer4": (1, 2048, 4, 4),
    "decoder.up1": (1, 256, 8, 8),
    "decoder.up2": (1, 128, 15, 15),
    "decoder.up3": (1, 64, 30, 30),
    "decoder.up4": (1, 32, 60, 60),
    "decoder.up5": (1, 16, 120, 120),
}

SCALE_FACTORS = {
    "encoder.layer1": 4, "encoder.layer2": 8, "encoder.layer3": 15, "encoder.layer4": 30,
    "decoder.up1": 15, "decoder.up2": 8, "decoder.up3": 4, "decoder.up4": 2, "decoder.up5": 1
}


"""
LMDB related utilities
"""
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

    # New column for the reference map key
    df["reference_map"] = df["patch_id"].apply(lambda x: x + "_reference_map")
    df = df[["s1_name", "reference_map", "split"]]

    # Split the data into train, validation, and test sets according to the "split" column
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    # Create lists of (image_key, reference_key) pairs
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

def get_image_and_mask_from_key(image_key, reference_key = None, lmdb_path=None):
    """
    Get the image and mask from the LMDB database based on the image key and reference key.

    Args:
        image_key (str): The image key.
        reference_key (str): The reference key.
        lmdb_path (str): Path to the LMDB database.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The image and mask.
    """
    env = lmdb.open(lmdb_path, readonly=True, lock = False)
    with env.begin() as txn:
        image_data = load(txn.get(image_key.encode()))
        if reference_key is not None:
            mask_data = load(txn.get(reference_key.encode()))
            mask = mask_data["Data"]
        else :
            mask = None

    # Get the image and mask
    image = np.stack([image_data[band] for band in ["VH", "VV"]])

    return image, mask

"""
Mask Functions
"""

# Function to replace pixel values with class indices
def replace_pixel_values_with_class_indices(mask, pixel_value_to_class_index = pixel_value_to_class_index):
    """
    Replaces pixel values with class indices in the mask.
    """ 
    class_indices_mask = np.zeros_like(mask, dtype=np.int32)

    # Iterate over the pixel values and class indices
    for value, index in pixel_value_to_class_index.items():
        class_indices_mask[mask == value] = index

    return class_indices_mask

"""
Visualization Functions
"""

def apply_color_map(mask, color_map = color_map):
    """
    Apply a color map to a mask.
    """
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_index, color in color_map.items():
        mask_rgb[mask == class_index] = color
    return mask_rgb

def display_from_image_and_mask(image, mask, color_map=color_map):
    """
    Displays a visualization of the image and mask.

    Args:
        image (np.ndarray): The image to display.
        mask (np.ndarray): The mask to display. Already in class indices.
        color_map (Dict[int, List[int]]): A dictionary mapping class indices to RGB colors.
    """
    mask_colored = apply_color_map(mask, color_map)

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the VH channel
    axes[0].imshow(image[0], cmap="gray")
    axes[0].set_title("VH Channel Visualization")
    axes[0].axis("off")

    # Display the mask
    axes[1].imshow(mask_colored)
    axes[1].set_title("Mask")
    axes[1].axis("off")
    plt.show()


def display_image_reference_inference(model, lmdb_path, ref_image, ref_reference, color_map = color_map, pixel_value_to_class_index = pixel_value_to_class_index, img_size = 120):
    """
    Displays a visualization of (handpicked) good and bad predictions of the model.

    Args:
        model (torch.nn.Module): The model to use for inference.
        lmdb_path (str): Path to the LMDB database.
        ref_image (str): The reference image key.
        ref_reference (str): The reference map key.
        color_map (Dict[int, List[int]]): A dictionary mapping class indices to RGB colors.
        pixel_value_to_class_index (Dict[int, int]): A dictionary mapping pixel values to class indices.

    """
        
    model.eval()
    env = lmdb.open(lmdb_path, readonly=True)
    
    with env.begin() as txn:
        # Load image data
        image_data = load(txn.get(ref_image.encode()))
        image = np.stack([image_data[band] for band in ["VH", "VV"]])
        
        # Convert image to PyTorch tensor and unsqueeze
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

        if img_size == 128:
            padded_img_tensor = pad_image(image_tensor, 128, 128)
        else:
            padded_img_tensor = pad_image(image_tensor, 120, 120)
        
        # Load reference map
        real_mask_data = load(txn.get(ref_reference.encode()))
        real_mask = real_mask_data["Data"]

        # Predict the mask
        pred = inference(padded_img_tensor, model)
        pred_mask_colored = apply_color_map(pred.squeeze(0).cpu(), color_map)

        real_mask_indices = replace_pixel_values_with_class_indices(real_mask, pixel_value_to_class_index)
        real_mask_colored = apply_color_map(real_mask_indices, color_map)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Display the VH channel
        axes[0].imshow(image[0], cmap="gray")
        axes[0].set_title(f"$\\bf{{VH\\ Channel}}$", fontsize=20)
        axes[0].axis("off")

        # Display the reference map
        axes[1].imshow(real_mask_colored)
        axes[1].set_title(f"$\\bf{{Original\\ Reference\\ Map}}$", fontsize=20)
        axes[1].axis("off")

        # Display the predicted mask
        axes[2].imshow(pred_mask_colored)
        axes[2].set_title(f"$\\bf{{Predicted\\ Mask}}$", fontsize=20)
        axes[2].axis("off")

        plt.show()

def get_reference_ref(image_ref, parquet_path):
    """
    Returns the corresponding reference map key for a given image reference.
    """
    df = pd.read_parquet(parquet_path)
    df["reference_map"] = df["patch_id"] + "_reference_map"
    lookup = dict(zip(df["s1_name"], df["reference_map"]))
    
    return lookup[image_ref]

def plot_query_examples(
    image_ref=None,
    mask_ref=None,
    region=None,
    layers_to_plot=None,
    model=None,
    dataset_lmdb_path=None,
    parquet_path=None,
    index_to_label=None,
    label_to_color=None,
    df_top_matches=None,
    top_x=5,
    target="infer",  # "infer" or "ref",
    filename=None,
    format="pdf",
    result_name="Example"
):
    x1, y1, x2, y2 = map(int, region)

    # Load query image and reference map
    img_grayscale_query, reference_map_query = get_image_and_mask_from_key(
        image_ref, mask_ref, lmdb_path=dataset_lmdb_path
    )
    reference_map_query = replace_pixel_values_with_class_indices(reference_map_query)
    reference_map_query_colored = apply_color_map(reference_map_query)
    unique_classes_ref = np.unique(reference_map_query[y1:y2, x1:x2]).tolist()

    # Run inference on query to get predicted mask
    mask_infer_indices_query = inference(img_grayscale_query, model).squeeze(0).cpu().numpy()
    query_mask_colored = apply_color_map(mask_infer_indices_query)
    unique_classes_query_infer = np.unique(mask_infer_indices_query[y1:y2, x1:x2]).tolist()

    query_img_grayscale = img_grayscale_query[0]

    max_matches = top_x
    n_layers = len(layers_to_plot)
    total_grid_rows = max(n_layers * 2, 3)
    total_grid_cols = 1 + 1 + max_matches

    column_widths = [1.5, 0.5] + [1.0] * max_matches
    height_ratios = [1.0] * total_grid_rows

    fig = plt.figure(figsize=(4 * total_grid_cols, 4 * total_grid_rows))
    grid = gridspec.GridSpec(
        total_grid_rows, total_grid_cols, figure=fig,
        width_ratios=column_widths, height_ratios=height_ratios
    )

    all_unique_labels = set(unique_classes_query_infer + unique_classes_ref)

    # Determine prototype cell size in figure coordinates
    pos_proto = grid[0, 2].get_position(fig)
    proto_w = pos_proto.width
    proto_h = pos_proto.height

    # Add small vertical gap between left‐column images
    vertical_gap = 0.05
    block_height = 3 * proto_h + 2 * vertical_gap
    block_y0 = 0.5 - (block_height / 2)

    # Center left plots horizontally within column 0
    pos_left = grid[0, 0].get_position(fig)
    left_x = pos_left.x0 + (pos_left.width - proto_w) / 2

    # Top: Full Query Image
    ax_query_orig = fig.add_axes([
        left_x,
        block_y0 + 2 * (proto_h + vertical_gap),
        proto_w,
        proto_h
    ])
    ax_query_orig.imshow(query_img_grayscale, cmap="gray")
    rect_qo = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="cyan", facecolor="none"
    )
    ax_query_orig.add_patch(rect_qo)
    ax_query_orig.set_title(f"$\\bf{{Full\\ Query\\ Image}}$", fontsize=16)
    ax_query_orig.axis("off")

    # Middle: Full Inferred Query
    ax_query_infer = fig.add_axes([
        left_x,
        block_y0 + 1 * (proto_h + vertical_gap),
        proto_w,
        proto_h
    ])
    ax_query_infer.imshow(query_mask_colored)
    rect_qi = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="cyan", facecolor="none"
    )
    ax_query_infer.add_patch(rect_qi)
    ax_query_infer.set_title(
        f"$\\bf{{Full\\ Inferred\\ Query}}$" + "\n" +
        f"Classes: {unique_classes_query_infer}", fontsize=16
    )
    ax_query_infer.axis("off")

    # Bottom: Original Reference Map
    ax_query_gt = fig.add_axes([
        left_x,
        block_y0 + 0 * (proto_h + vertical_gap),
        proto_w,
        proto_h
    ])
    ax_query_gt.imshow(reference_map_query_colored)
    rect_qg = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="cyan", facecolor="none"
    )
    ax_query_gt.add_patch(rect_qg)
    ax_query_gt.set_title(
        f"$\\bf{{Original\\ Reference\\ Map}}$" + "\n" +
        f"Classes: {unique_classes_ref}", fontsize=16
    )
    ax_query_gt.axis("off")

    # Turn off axes
    prototype_axes = np.empty((total_grid_rows, max_matches), dtype=object)
    for row_idx in range(total_grid_rows):
        for col_offset in range(max_matches):
            ax = fig.add_subplot(grid[row_idx, 2 + col_offset])
            ax.axis("off")
            prototype_axes[row_idx, col_offset] = ax

    # Plot prototypes (image + chosen mask)
    for layer_idx, layer_name in enumerate(layers_to_plot):
        img_row = layer_idx * 2
        mask_row = layer_idx * 2 + 1

        # vertical layer name
        fig.canvas.draw()
        ax_for_text = prototype_axes[img_row, 0]
        y_center_text = (
            ax_for_text.get_position().y0 +
            prototype_axes[mask_row, 0].get_position().y1
        ) / 2
        fig.text(
            ax_for_text.get_position().x0 - 0.02,
            y_center_text,
            layer_name,
            va="center", ha="right", rotation="vertical",
            fontsize=14, fontweight="bold"
        )

        # Fetch top‐X matches for this layer
        filtered = df_top_matches[
            (df_top_matches["image_id"] == image_ref) &
            (df_top_matches["original_region"] == tuple(region)) &
            (df_top_matches["layer"] == layer_name)
        ]
        matches_raw = ast.literal_eval(str(filtered["top_n"].values[0]))
        matches = matches_raw[:top_x] if isinstance(matches_raw, list) else []

        for proto_col_offset, match in enumerate(matches):
            match_score, match_ref, match_region_raw = match
            match_region = (
                tuple(map(int, match_region_raw))
                if not isinstance(match_region_raw, tuple)
                else match_region_raw
            )
            px1, py1, px2, py2 = match_region

            # Load prototype image + reference map
            prototype_ref = get_reference_ref(match_ref, parquet_path)
            img_proto, proto_refmap = get_image_and_mask_from_key(
                match_ref, prototype_ref, lmdb_path=dataset_lmdb_path
            )
            proto_refmap = replace_pixel_values_with_class_indices(proto_refmap)
            proto_refmap_colored = apply_color_map(proto_refmap)

            mask_infer_proto = inference(img_proto, model).squeeze(0).cpu().numpy()
            proto_mask_colored = apply_color_map(mask_infer_proto)
            proto_img_full = img_proto[0]

            if target == "infer":
                plot_map = proto_mask_colored
                unique_proto = np.unique(mask_infer_proto[py1:py2, px1:px2]).tolist()
                title_label = "Inferred\\ Mask"
            else:
                plot_map = proto_refmap_colored
                unique_proto = np.unique(proto_refmap[py1:py2, px1:px2]).tolist()
                title_label = "Reference\\ Map"
            all_unique_labels.update(unique_proto)

            # Prototype original image
            ax_po = prototype_axes[img_row, proto_col_offset]
            ax_po.imshow(proto_img_full, cmap="gray")
            rect_po = patches.Rectangle(
                (px1, py1), px2 - px1, py2 - py1,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax_po.add_patch(rect_po)
            ax_po.set_title(
                f"$\\bf{{{result_name}\\ {proto_col_offset + 1}}}$" + "\n" +
                f"Score: {round(match_score, 2)}", fontsize=16
            )

            # Prototype inferred or reference mask
            ax_pm = prototype_axes[mask_row, proto_col_offset]
            ax_pm.imshow(plot_map)
            rect_pm = patches.Rectangle(
                (px1, py1), px2 - px1, py2 - py1,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax_pm.add_patch(rect_pm)
            ax_pm.set_title(
                f"$\\bf{{{title_label}\\ {proto_col_offset + 1}}}$" + "\n" +
                f"Classes: {unique_proto}", fontsize=16
            )

    # Legend
    legend_handles = []
    for idx in sorted(all_unique_labels):
        label = index_to_label.get(idx)
        if label is not None:
            rgb = label_to_color.get(label, (0, 0, 0))
            color_norm = tuple(v / 255.0 for v in rgb)
            legend_handles.append(
                patches.Patch(facecolor=color_norm, edgecolor="black", label=f"{idx}: {label}")
            )

    last_row_y0 = prototype_axes[(n_layers - 1) * 2 + 1, 0].get_position().y0
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.6, last_row_y0),
        ncol=3,
        fontsize=12,
        frameon=False
    )

    grid.update(wspace=0.1, hspace=0.25)
    plt.tight_layout()
    plt.show()

    if filename:
        fig.savefig(f"{filename}.{format}", format=format, bbox_inches='tight')
        print(f"Figure saved as {filename}")

"""
Model Factories
"""
def create_base_model_skipconn(backbone ="resnet50", weights = None, in_channel = 2, num_classes = 20):
    model = CustomUnetSkipConn(
        encoder_name= backbone,   
        encoder_weights=weights,   
        in_channels=in_channel,    
        classes=num_classes,        
        activation="softmax",      
    )
    return model

def load_from_checkpoint_skipconn(checkpoint_path, num_classes= 20):
    model = create_base_model_skipconn(num_classes = num_classes)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(torch.load(checkpoint_path, weights_only = True), strict=False)
    return model

def load_base_with_bigearth_pretrained_skipconn(num_classes= 20):
    # Pretrained model
    model = create_base_model_skipconn(num_classes = num_classes)
    model_bigearth_classifier = BigEarthNetv2_0_ImageClassifier.from_pretrained(
        "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.1.1"
    )
    pretrained_weights = model_bigearth_classifier.state_dict()

    # Mapping from the pretrained to untrained model
    key_mapping = {
        pretrained_key: pretrained_key.replace("model.vision_encoder", "encoder")
        for pretrained_key in pretrained_weights.keys()
        if pretrained_key.startswith("model.vision_encoder")
    }

    # Map weights
    mapped_state_dict = {
        untrained_key: pretrained_weights[pretrained_key]
        for pretrained_key, untrained_key in key_mapping.items()
    }
    # Load model
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    return model

"""
U-Net
"""
class CustomDecoderSkipConn(nn.Module):
    def __init__(self, in_channels, decoder_channels):
        super().__init__()

        # Upsampling blocks with transposed convolution + ConvBlock
        def up_block(in_ch, out_ch, scale_factor):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.up1 = up_block(in_channels, decoder_channels[0], scale_factor=8/4)     # 4 → 8
        self.up2 = up_block(decoder_channels[0] + 1024, decoder_channels[1], scale_factor=15/8)  # Concatenate with encoder output (8 → 15)
        self.up3 = up_block(decoder_channels[1] + 512, decoder_channels[2], scale_factor=30/15)  # 15 → 30
        self.up4 = up_block(decoder_channels[2] + 256, decoder_channels[3], scale_factor=60/30)  # 30 → 60
        self.up5 = up_block(decoder_channels[3] + 64, decoder_channels[4], scale_factor=120/60)  # 60 → 120
        
    def forward(self, x, skip_connections):
        # print("skip_connections shape:", [f.shape for f in skip_connections])
        x = self.up1(x)
        x = torch.cat([x, skip_connections[4]], dim=1)  # Concatenate with the 4th encoder output
        x = self.up2(x)
        x = torch.cat([x, skip_connections[3]], dim=1)  # Concatenate with the 3rd encoder output
        x = self.up3(x)
        x = torch.cat([x, skip_connections[2]], dim=1)  # Concatenate with the 2nd encoder output
        x = self.up4(x)
        x = torch.cat([x, skip_connections[1]], dim=1)  # Concatenate with the 1st encoder output
        x = self.up5(x)
        return x

class CustomUnetSkipConn(smp.Unet):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights=None,
        decoder_channels=(256, 128, 64, 32, 16),
        in_channels: int = 2,
        classes: int = 20,
        activation="softmax",
    ):
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, activation=activation)

        self.decoder = CustomDecoderSkipConn(in_channels=self.encoder.out_channels[-1], decoder_channels=decoder_channels)

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        print("CustomUnet initialized with encoder channels:", self.encoder.out_channels)

    def forward(self, x):
        features = self.encoder(x)  # encoder features
        x = features[-1]  # Last encoder layer
        skip_connections = features[:-1] 

        # Pass feature map
        x = self.decoder(x, skip_connections)
        x = self.segmentation_head(x)
        return x

"""
Training and Inference Utilities
"""
def training(model,
             epoch_start,
             epoch_end,
             train_loader,
             val_loader,
             num_classes,
             lr=1e-4,
             model_name="unet120",
             freeze_epochs=2,
             loss_weights=(1.0, 1.0),
             save_dir = "../models/",
             ignore_index=20):
    """
    Training loop with combined Focal + Dice loss, per-class IoU/F1 logging, and tqdm progress.
    """
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Instantiate losses
    focal_loss = FocalLoss(mode="multiclass", ignore_index=ignore_index)  
    dice_loss = DiceLoss(mode="multiclass", ignore_index=ignore_index)  
    w_focal, w_dice = loss_weights
    

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    scaler = GradScaler()  # Mixed precision
    best_val_loss = float("inf")
    patience = 5  # Early stopping patience
    patience_counter = 0

    for epoch in range(epoch_start, epoch_end + 1):
        if epoch == freeze_epochs + 1:
            for param in model.encoder.parameters():
                param.requires_grad = True
            print(f"Unfroze encoder at epoch {epoch}")

        # Training
        model.train()
        train_loss, train_iou, train_f1 = 0.0, 0.0, 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{epoch_end}", unit="batch")
        num_batches = 0

        for images, masks in train_loader_tqdm:
            images, masks = images.to(device), masks.to(device)
            masks = masks.argmax(dim=1)  # Convert one-hot to class indices
            # print(f"Image shape: {images.shape}, Mask shape: {masks.shape}")

            optimizer.zero_grad()
            with autocast(device_type=device_type):
                outputs = model(images)
                #print(f"Output shape: {outputs.shape}, Mask shape: {masks.shape}")
                loss_f = focal_loss(outputs, masks)
                loss_d = dice_loss(outputs, masks)
                loss = w_focal * loss_f + w_dice * loss_d

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()

            # Compute IoU metrics
            tp, fp, fn, tn = smp.metrics.get_stats(
                outputs.argmax(dim=1).to(torch.int32), masks, mode="multiclass", num_classes=num_classes
            )
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

            train_loss += loss.item()
            train_iou += iou.item()
            train_f1 += f1.item()
            
            num_batches += 1
            avg_loss = train_loss / num_batches
            avg_iou = train_iou / num_batches
            avg_f1 = train_f1 / num_batches
            
            train_loader_tqdm.set_postfix(loss=avg_loss, iou=avg_iou, f1=avg_f1, lr=optimizer.param_groups[0]["lr"])


        # Log Training
        epoch_loss_train = train_loss / len(train_loader)
        epoch_iou_train = train_iou / len(train_loader)
        epoch_f1_train = train_f1 / len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss_train, epoch)
        writer.add_scalar("IoU/train", epoch_iou_train, epoch)
        writer.add_scalar("F1/train", epoch_f1_train, epoch)
        print(f"Epoch {epoch}, Train Loss: {epoch_loss_train:.4f}, IoU: {epoch_iou_train:.4f}, F1: {epoch_f1_train:.4f}")

        # Validation
        model.eval()
        val_loss, val_iou, val_f1 = 0.0, 0.0, 0.0
        val_loader_tqdm = tqdm(val_loader, desc="Validation", unit="batch")
        num_batches = 0

        with torch.no_grad():
            for images, masks in val_loader_tqdm:
                images, masks = images.to(device), masks.to(device)
                masks = masks.argmax(dim=1)
                
                with autocast(device_type=device_type):
                    outputs = model(images)
                    loss_f = focal_loss(outputs, masks)
                    loss_d = dice_loss(outputs, masks)
                    loss = w_focal * loss_f + w_dice * loss_d

                tp, fp, fn, tn = smp.metrics.get_stats(
                    outputs.argmax(dim=1).to(torch.int32), masks, mode="multiclass", num_classes=num_classes
                )
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

                val_loss += loss.item()
                val_iou += iou.item()
                val_f1 += f1.item()

                num_batches += 1
                avg_loss = val_loss / num_batches
                avg_iou = val_iou / num_batches
                avg_f1 = val_f1 / num_batches

                val_loader_tqdm.set_postfix(loss=avg_loss, iou=avg_iou, f1=avg_f1, lr=optimizer.param_groups[0]["lr"])

        # Log Validation
        epoch_loss_val = val_loss / len(val_loader)
        epoch_iou_val = val_iou / len(val_loader)
        epoch_f1_val = val_f1 / len(val_loader)
        writer.add_scalar("Loss/val", epoch_loss_val, epoch)
        writer.add_scalar("IoU/val", epoch_iou_val, epoch)
        writer.add_scalar("F1/val", epoch_f1_val, epoch)
        print(f"Epoch {epoch}, Val Loss: {epoch_loss_val:.4f}, IoU: {epoch_iou_val:.4f}, F1: {epoch_f1_val:.4f}")

        # LR scheduler & early stopping
        scheduler.step(epoch_loss_val)
        print(f" Learning rate after epoch {epoch}: {optimizer.param_groups[0]['lr']}")
        torch.save(model.state_dict(), f"{save_dir}/{model_name}_epoch_{epoch}.pth")
        if epoch_loss_val < best_val_loss:
            best_val_loss = epoch_loss_val
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/{model_name}_best.pth")  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # Print last used learning rate
    last_lr = optimizer.param_groups[0]["lr"]
    print(f"Last used learning rate: {last_lr}")

    writer.flush()
    writer.close()
    return model, optimizer

def calculate_scores(model, test_loader, device, num_classes, ignore_index=20):
    """
    Perform evaluation on a given test set and calculate loss, IoU, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    test_f1 = 0.0
    criterion = FocalLoss(mode="multiclass", ignore_index=ignore_index)  

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Calculating scores"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Convert one-hot encoded masks to class indices
            masks = masks.argmax(dim=1)
            
            loss = criterion(outputs, masks)
            tp, fp, fn, tn = smp.metrics.get_stats(
                    outputs.argmax(dim=1).to(torch.int32), masks, mode="multiclass", num_classes=num_classes
                )
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

            test_loss += loss.item()
            test_iou += iou.item()
            test_f1 += f1.item()

    test_loss /= len(test_loader)
    test_iou /= len(test_loader)
    test_f1 /= len(test_loader)

    print(f"Test Loss: {test_loss}, Test IoU: {test_iou}, Test F1: {test_f1}")
    return test_loss, test_iou, test_f1

def inference(img, model):
    """
    Perform inference on a single image.
    """
    # if image not type torch tensor
    if not isinstance(img, torch.Tensor):	
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension	
    img = img.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1)
    return pred

"""
Dataset and DataLoader Utilities
"""
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
        mask_indices_one_hot = F.one_hot(torch.tensor(mask_indices).long(), num_classes=self.num_classes).permute(2, 0, 1).float().numpy() # (H, W, C) -> (C, H, W)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image_tensor, mask=mask_indices_one_hot)
            image_tensor = augmented["image"]
            mask_indices_one_hot = augmented["mask"]

        # Convert to tensors
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
        mask_indices_one_hot = torch.tensor(mask_indices_one_hot, dtype=torch.long)

        return image_tensor, mask_indices_one_hot


"""
Activation LMDB Utilities
"""

def store_activations(matches, model, lmdb_path, source_lmdb_path, layer_names, break_flag, map_size=2*10**10):
    """
    Store activations from the model into an LMDB database.
    """
    print("Storing activations in LMDB at path:", lmdb_path)

    # Create LMDB env
    env = lmdb.open(lmdb_path, map_size=map_size)
    source_env = lmdb.open(source_lmdb_path, readonly=True)

    activations = {}
    setup_hooks(model, activations)

    # Iterate over matches and store activations    
    for idx, match in tqdm(enumerate(matches), total=len(matches)):
        if break_flag: 
            break
        image_key, reference_key = match

        with source_env.begin() as txn:
            image_data = load(txn.get(image_key.encode()))
            image_bands = ["VH", "VV"] # Sentinel 1 bands
            image_tensor = np.stack([image_data[band] for band in image_bands])
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0).to(device)

        # Forward, capture activations
        activations.clear()
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # Store activations in LMDB
        batch_activations = {k: v.cpu().numpy() for k, v in activations.items()}
        with env.begin(write=True) as txn:
            for layer_name, activation in batch_activations.items():
                if layer_name not in layer_names:
                    continue
                # Create a unique key for each layer"s activation   
                layer_key = f"{image_key}_{layer_name}"
                txn.put(layer_key.encode(), activation.tobytes())

    env.close()
    source_env.close()


def test_store_activations(model, lmdb_path, source_lmdb, image_reference, layer_name):
    """
    Verify that activations are stored correctly in LMDB.
    """
    activations = {}
    setup_hooks(model, activations)

    env = lmdb.open(lmdb_path, readonly=True)
    activation_tensor = load_activation(image_reference, layer_name, env = env)

    if activation_tensor is not None:
        print("Loaded activation shape:", activation_tensor.shape)
    else: 
        print("Activation not found")
    env.close()

    image, mask = get_image_and_mask_from_key(image_reference, lmdb_path = source_lmdb)
    image = torch.tensor(image, dtype=torch.float32).to(device).unsqueeze(0)

    # Forward, capture activation
    output = model(image)
    layer_activations = activations[layer_name]

    # Compare stored activation with computed activation
    if torch.allclose(layer_activations, activation_tensor.to(device)):
        print("Activations match successfully!")


def load_activation(image_reference, layer_name, env):
    """
    Load the activation tensor from LMDB.
    """
    with env.begin() as txn:
        key = f"{image_reference}_{layer_name}".encode()
        activation_data = txn.get(key)
        if activation_data is None:
            return None

        activation_array = np.frombuffer(activation_data, dtype=np.float32)
        activation_tensor = torch.tensor( activation_array.copy(), dtype=torch.float32 ).reshape(LAYER_SHAPES[layer_name])
        return activation_tensor


def extract_region_activations(activation, region_coords, layer_name):
    """
    Extracts a specific region from the activation tensor based on the provided coordinates and layer name.
    """
    factor = SCALE_FACTORS.get(layer_name)
    if factor is None:
        raise ValueError(f"Layer {layer_name} not found in SCALE_FACTORS.")

    x1, y1, x2, y2 = region_coords
    # Scale coordinates based on the factor
    scaled_x1 = int(np.round(x1 / factor))
    scaled_y1 = int(np.round(y1 / factor))
    scaled_x2 = int(np.ceil(x2 / factor))
    scaled_y2 = int(np.ceil(y2 / factor))

    extracted = activation[:, :, scaled_y1:scaled_y2, scaled_x1:scaled_x2]
    return extracted

def get_activation(name, dictionary):
    def hook(model, input, output):
        dictionary[name] = output.detach()  # Capture the output and detach from the computation graph
    return hook

def setup_hooks(model, activations_dict):
    """
    Set up hooks to capture activations from specific layers of the model.

    Note: Specific for the current implementation of the model.
    """
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Sequential) and (name.startswith("encoder") or name.startswith("decoder")) and "downsample" not in name:
            layer.register_forward_hook(get_activation(name, activations_dict))

"""
Similarity Utilities
"""
def compute_similarity(query_activation_flat, candidate_activation_flat, metric="cosine"):
    """
    Compute similarity efficiently using pre-flattened tensors.

    Note: Currently supports "cosine" and "euclidean" metrics.
    """
    return F.cosine_similarity(query_activation_flat, candidate_activation_flat, dim=1).item() if metric == "cosine" else -torch.norm(query_activation_flat - candidate_activation_flat, dim=1).item()

def plot_similarities(query_image, query_predicted, similar_images, similar_masks, titles, draw_boxes):
    """
    Displays the query image, predicted mask, and similar images with their masks.

    Args:
        query_image (torch.Tensor): The query image tensor in (C, H, W) format.
        query_predicted (np.ndarray): The predicted mask for the query image in (H, W) format.
        similar_images (List[torch.Tensor]): A list of similar image tensors.
        similar_masks (List[np.ndarray]): A list of masks for the similar images.
        titles (List[str]): A list of titles for the similar images.
    """
    num_images = len(similar_images) + 1  # Query image + similar images
    fig, axes = plt.subplots(2, num_images, figsize=(18, 6))

    # Display Query Image and Mask
    axes[0, 0].imshow(query_image[1].cpu().numpy(), cmap="gray")  # First Band
    axes[0, 0].set_title("Query Image")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(query_predicted)  # Query mask (HW)
    axes[1, 0].set_title("Predicted Mask")
    axes[1, 0].axis("off")

    # Draw box in query image
    x1,y1,x2,y2 = draw_boxes.pop(0)
    rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="c", facecolor="none")
    rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="c", facecolor="none")
    axes[0, 0].add_patch(rect1)
    axes[1, 0].add_patch(rect2)

    # Display Top-5 Similar Images and Masks
    for i, (img, mask, title, draw_box) in enumerate(zip(similar_images, similar_masks, titles, draw_boxes), start=1):
        axes[0, i].imshow(img[0].cpu().numpy(), cmap="gray")
        axes[0, i].set_title(f"Example {i}: {title}")
        axes[0, i].axis("off")
        axes[1, i].imshow(mask)  # Mask (HW)
        axes[1, i].set_title(f"Segmentation Mask {i}")
        axes[1, i].axis("off")

        # Draw box if available
        if draw_box:
            x1, y1, x2, y2 = draw_box
            rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
            rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
            axes[0, i].add_patch(rect1)
            axes[1, i].add_patch(rect2)

    plt.tight_layout()
    plt.show()

def find_n_similar_regions(query_activation, layer_name, train_image_keys, activation_lmdb_path, 
                           n=5, region_coords=None, metric="cosine"):
    """
    Find the top-N most similar regions in training images compared to a fixed region in the query image.
    
    - Uses a sliding window to extract candidate regions from training images.
    - Computes similarity between query region and each candidate region.
    - Maintains a heap of the top-N most similar regions.
    
    Args:
        query_activation (np.ndarray): Feature activation for the query image.
        layer_name (str): The model layer to extract activations from.
        train_image_keys (List[str]): List of training image keys.
        activation_lmdb_path (str): Path to the LMDB database storing activations.
        n (int): Number of most similar regions to return.
        region_coords (Tuple[int, int, int, int]): (x1, y1, x2, y2) coordinates of the query region.
        metric (str): Similarity metric (e.g., "cosine", "euclidean").

    Returns:
        List[Tuple[float, str, Tuple[int, int, int, int]]]: 
            Sorted list of (similarity_score, image_key, region_coords).
    """
    min_heap = []
    env = lmdb.open(activation_lmdb_path, readonly=True, lock=False)

    # Extract query activation for the specified region
    query_tensor = query_activation.to(device)

    if region_coords:
        query_tensor = extract_region_activations(query_tensor, region_coords, layer_name)

    query_tensor_flat = query_tensor.flatten(start_dim=1)

    win_w, win_h = query_tensor.shape[-2:]
    acti_w, acti_h = query_activation.shape[-2:]

    # Amount of division for sliding window
    divisions_w = int(np.ceil(acti_w / win_w))
    divisions_h = int(np.ceil(acti_h / win_h))

    factor = SCALE_FACTORS[layer_name]

    # Debugging information
    if divisions_w == 0 or divisions_h == 0:
        print("Error: Query activation map is not square.")
        print(f"Activation Map Dimensions: {acti_w}x{acti_h}, Window Dimensions: {win_w}x{win_h}")
        print(f"temp_query shape: {query_tensor.shape}")
        print(f"query shape: {query_activation.shape}")
        print(f"region_coords: {region_coords}")
        return []

    with env.begin() as txn:
        for train_image_key in train_image_keys:
            train_activation = load_activation(train_image_key, layer_name, env=env)
            if train_activation is None:
                continue

            train_tensor = train_activation.to(device)
            _, _, H, W = train_tensor.shape

            x_positions = list(range(0, W, win_w))
            if divisions_w * win_w > W:
                # print("Warning: Activation map is smaller than query window.")
                x_positions[-1] = W - win_w
            
            y_positions = list(range(0, H, win_h))
            if divisions_h * win_h > H:
                # print("Warning: Activation map is smaller than query window.")
                y_positions[-1] = H - win_h

            # Slide a window over the activation map
            for y in y_positions:
                for x in x_positions:
                    candidate_activation = train_tensor[:, :, y:y+win_h, x:x+win_w]
                    candidate_activation_flat = candidate_activation.flatten(start_dim=1)
                    similarity = compute_similarity(query_tensor_flat, candidate_activation_flat, metric=metric)

                    # Store top-N matches
                    region = (x*factor, y*factor, (x+win_w)*factor,(y+win_h)*factor)
                    heapq.heappush(min_heap, (similarity, train_image_key, region))
                    if len(min_heap) > n:
                        heapq.heappop(min_heap)

    env.close()
    return sorted(min_heap, key=lambda x: x[0], reverse=True)  # Sort by highest similarity

def find_n_similar_images(query_image, layer_names, image_keys, 
                             activations_lmdb_path, images_lmdb_path, model, 
                             img_hw=(120,120), color_map=color_map, region_coords=None, n = 5, plotting = True):
    """
    Displays the query image alongside the most similar image regions.

    - Extracts the query region.
    - Finds top matching regions using sliding window search.
    - Displays retrieved regions from training images.

    Args:
        query_image (np.ndarray): The query image to search for similar regions.
        layer_names (str or List[str]): Layer names to investiage.
        image_keys (List[str]): List of training image keys. 
        activations_lmdb_path (str): Path to the LMDB database storing activations.
        images_lmdb_path (str): Path to the LMDB database storing images. For plotting.
        model (torch.nn.Module): The model used for inference.
        img_hw (Tuple[int, int]): Height and width of the input images.
        color_map (np.ndarray): Color map for visualization.
        region_coords (Tuple[int, int, int, int]): Coordinates of the query region.
        n (int): Number of similar regions to return.
        plotting (bool): Whether to plot the results or return them.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activations = {}
    setup_hooks(model, activations)
    activations.clear()
    
    if not isinstance(layer_names, list): layer_names = [layer_names]

    # Get the query mask
    query_image = torch.tensor(query_image, dtype=torch.float32).to(device)
    query_image = pad_image(query_image.unsqueeze(0), img_hw[0], img_hw[1])

    model.eval()
    output = inference(query_image, model)
    out_mask_colored = apply_color_map(output.squeeze(0).cpu(), color_map)
    
    query_activations = [activations[layer_name] for layer_name in layer_names]

    # Find similar images based on activations
    top_n_results = [find_n_similar_regions(query_activations, layer_name, image_keys, activations_lmdb_path, n=50, region_coords=region_coords) for layer_name, query_activations in zip(layer_names, query_activations)]

    if not plotting: return top_n_results

    for results, layer_name in zip(top_n_results, layer_names):
        print(f"   Results for layer: {layer_name}")
        print(f"   Top N Results: {results}")
        similar_images, similar_masks, titles, draw_boxes = [], [], [], []
        draw_boxes.append(region_coords) # Draw the query region as a box
    
        for similarity_score, image_key, region_descriptor in results[:n]:
            # Load the images
            image, _ = get_image_and_mask_from_key(image_key, lmdb_path=images_lmdb_path)
            image = torch.tensor(image, dtype=torch.float32).to(device)
            image = pad_image(image.unsqueeze(0), img_hw[0], img_hw[1])
            # Get the predicted mask
            pred_mask = model(image).argmax(dim=1).squeeze(0).cpu()
            pred_mask = apply_color_map(pred_mask, color_map)

            # Append results
            similar_images.append(image.squeeze(0)) # Remove batch dimension
            similar_masks.append(pred_mask)
            titles.append(f"Sim: {similarity_score:.3f}\nRegion: {region_descriptor}")

            draw_boxes.append(region_descriptor)

        plot_similarities(
            query_image=query_image.squeeze(0), 
            query_predicted=out_mask_colored, 
            similar_images=similar_images, 
            similar_masks=similar_masks, 
            titles=titles,
            draw_boxes=draw_boxes
        )
    return top_n_results

"""
General Experiment Utilities
"""

"""
Overlap
"""
def get_top_subset(layer, top_n):
    """
    Return a set of unique keys for the first top_n entries of the layer.
    """
    return {(name, region) for (score, name, region) in layer[:top_n]}

def get_pairwise_overlap(list_of_lists, layer_names, eta = 0.85):
    """
    Compute pairwise overlaps between Top-{5,10,20,50} elements of each layer in the list_of_lists.
    """
    results = {}
    n = len(layer_names)
    # Precompute sets for each layer:
    top_5_sets = [get_top_subset(layer, 5) for layer in list_of_lists]
    top_10_sets = [get_top_subset(layer, 10) for layer in list_of_lists]
    top_20_sets = [get_top_subset(layer, 20) for layer in list_of_lists]
    top_50_sets = [get_top_subset(layer, 50) for layer in list_of_lists]
    
    # Compare each pair (i, j)
    for i in range(n):
        for j in range(i + 1, n):
            overlap_5 = set()
            overlap_10 = set()
            overlap_20 = set()
            overlap_50 = set()

            for ref1, region1 in top_5_sets[i]:
                for ref2, region2 in top_5_sets[j]:
                    if ref1 == ref2 and regions_match_via_iou(region1, region2, eta):
                        overlap_5.add((ref1, region1))
            
            for ref1, region1 in top_10_sets[i]:
                for ref2, region2 in top_10_sets[j]:
                    if ref1 == ref2 and regions_match_via_iou(region1, region2, eta):
                        overlap_10.add((ref1, region1))
            
            for ref1, region1 in top_20_sets[i]:
                for ref2, region2 in top_20_sets[j]:
                    if ref1 == ref2 and regions_match_via_iou(region1, region2, eta):
                        overlap_20.add((ref1, region1))

            for ref1, region1 in top_50_sets[i]:
                for ref2, region2 in top_50_sets[j]:
                    if ref1 == ref2 and regions_match_via_iou(region1, region2, eta):
                        overlap_50.add((ref1, region1))

            results[(i, j)] = {
                "First Layer": layer_names[i],
                "Second Layer": layer_names[j],
                "top 5": len(overlap_5),
                "top 10": len(overlap_10),
                "top 20": len(overlap_20),
                "top 50": len(overlap_50),
                "overlap top 5": overlap_5,
                "overlap top 10": overlap_10,
                "overlap top 20": overlap_20,
                "overlap top 50": overlap_50
            }
    return results

def regions_match_via_iou(r1, r2, eta=0.7):
    """
    Returns True if IoU between r1 and r2 is above eta
    """
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[2], r2[2])
    y2 = min(r1[3], r2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
    area2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    # print(f"iou: {iou}, eta: {eta}")
    return iou >= eta


def get_aggregate_overlaps(all_image_layers, layer_names, eta = 0.85):
    """
    Aggregate pairwise overlaps of Top-{5,10,20,50} across all image layers.
    
    Args:
        overlap_results (dict): Dictionary with pairwise overlaps.

    Returns a dictionary with aggregated results.
    """
    aggregated = {}
    for image_layers in all_image_layers:
        # Compute overlaps for current image.
        results = get_pairwise_overlap(image_layers, layer_names, eta)
        for key, value in results.items():
            # key is (i, j) corresponding to the layer indices.
            if key not in aggregated:
                aggregated[key] = {
                    "First Layer": value["First Layer"],
                    "Second Layer": value["Second Layer"],
                    "top 5": [],
                    "top 10": [],
                    "top 20": [],
                    "top 50": []
                }
            for metric in ["top 5", "top 10", "top 20", "top 50"]:
                aggregated[key][metric].append(value[metric])

    # Average overlap counts for each layer pair.
    averaged = {}
    for key, value in aggregated.items():
        averaged[key] = {
            "First Layer": value["First Layer"],
            "Second Layer": value["Second Layer"],
            "top 5": np.mean(value["top 5"]),
            "top 10": np.mean(value["top 10"]),
            "top 20": np.mean(value["top 20"]),
            "top 50": np.mean(value["top 50"]),
        }

    return averaged

""" 
Multi-Purpose Utilities
"""

def pad_image(img_tensor, target_height, target_width):
    """
    Pads an image tensor to the target height and width.
    """
    _, _, h, w = img_tensor.shape
    pad_h = target_height - h
    pad_w = target_width - w
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    padded_img = F.pad(img_tensor, padding, mode="constant", value=0)
    return padded_img

def show_overlap_matrix(data, targets, layer_names, mode = "overlap", title_in = None, save_plot = False, result_name="Example"):
    """
    Visualize the overlap matrix for different layers and targets.

    Args:
        data (Dict): Dictionary containing the overlap data.
        targets (List[str]): List of target names (e.g. "top 5")
        layer_names (List[str]): List of layer names (e.g. "encoder.layer3")
    """
    if not isinstance(layer_names, list): layer_names = [layer_names]
    if not isinstance(targets, list): targets = [targets]
    
    # Loop over all targets
    for target in targets:
        matrix = np.zeros((len(layer_names), len(layer_names)))
        for i in range(len(layer_names)):
            for j in range(len(layer_names)):
                if i != j:
                    # Get overlap count for target
                    if (i, j) in data.keys():
                        matrix[i,j] = data[(i, j)][target]
                    elif (j, i) in data.keys():
                        matrix[i,j] = data[(j, i)][target]
                    else:
                        matrix[i,j] = 0 # when i = j
                    
        # Convert matrix to DataFrame
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.colorbar(ax.matshow(matrix, cmap="Blues"))

        # Annotate heatmap
        for i in range(len(layer_names)):
            for j in range(len(layer_names)):
                ax.text(j, i, round(matrix[i, j],2), ha="center", va="center", color="black")

        if title_in is None:
            if mode == "overlap":
                print(f"Overlap Matrix - {target}")
                title = f"{result_name} Overlap Matrix - {target}"
            else:
                (p1,p2) = data[(0,1)][f"{result_name} Overlap Top X with Top Y"]
                title = f"Top {p1} {result_name} Overlap with Top {p2}"
        else:
            title = title_in

        ax.set_xticks(range(len(layer_names)))
        ax.set_yticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=30)
        ax.set_yticklabels(layer_names)
        plt.title(title)
        plt.xlabel("Second Layer")
        plt.ylabel("First Layer")
        fig.tight_layout()

        if save_plot:
            save_path = f"plots/overlaps_{target}"
            fig.savefig(save_path, bbox_inches="tight")

        plt.show()

            