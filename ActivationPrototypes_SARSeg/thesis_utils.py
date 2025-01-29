'''
Utilities for SAR image segmentation
Authored by Janes Sanne
'''

'''
Imports
'''
# Standard Libraries
import random
import re

# Plotting
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# Machine Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.base import SegmentationHead
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



''' 
Basic Parameters
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

color_map = {
    1: [255, 0, 0],       # Urban fabric (Red)
    2: [255, 69, 0],      # Industrial or commercial units (Orange-Red)
    3: [255, 165, 0],     # Arable land (Orange)
    4: [255, 255, 0],     # Permanent crops (Yellow)
    5: [173, 255, 47],    # Pastures (Green-Yellow)
    6: [34, 139, 34],     # Complex cultivation patterns (Forest Green)
    7: [192, 192, 192],   # Land principally occupied by agriculture, with significant areas of natural vegetation (Silver)
    8: [128, 128, 0],     # Agro-forestry areas (Olive)
    9: [0, 100, 0],       # Broad-leaved forest (Dark Green)
    10: [0, 128, 0],      # Coniferous forest (Green)
    11: [34, 139, 34],    # Mixed forest (Forest Green)
    12: [154, 205, 50],   # Natural grassland and sparsely vegetated areas (Yellow-Green)
    13: [107, 142, 35],   # Moors, heathland and sclerophyllous vegetation (Olive Drab)
    14: [255, 255, 224],  # Transitional woodland, shrub (Light Yellow)
    15: [210, 180, 140],  # Beaches, dunes, sands (Tan)
    16: [0, 255, 255],    # Inland wetlands (Cyan)
    17: [0, 191, 255],    # Coastal wetlands (Deep Sky Blue)
    18: [0, 0, 255],      # Inland waters (Blue)
    19: [25, 25, 112],    # Marine waters (Midnight Blue)
    20: [255, 255, 255],  # Unlabeled (White)
}

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

# According to https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf we can match pixel values to classes
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

'''
LMDB related utilities
'''

# FIXME: Remove this function
def match_keys_dynamic(lmdb_path):
    """
    OUTDATED!!!!!!!!!!!!!!!!!!!!!!!
    Matches image keys with their corresponding reference map keys in the LMDB database
    based on dynamically extracted shared portions.

    Args:
        lmdb_path (str): Path to the LMDB database.

    Returns:
        List[Tuple[str, str]]: A list of (image_key, reference_key) pairs.
    """
    matches = []
    env = lmdb.open(lmdb_path, readonly=True, lock = False)
    with env.begin() as txn:
        cursor = txn.cursor()
        image_keys = {}
        reference_keys = {}

        for key, _ in cursor:
            key_str = key.decode()  # Decode bytes to string

            # Extract the shared index for image and reference keys dynamically
            image_match = re.search(r"([A-Z0-9]{5})_(\d+_\d+)$", key_str)
            reference_match = re.search(r"([A-Z0-9]{5})_(\d+_\d+)_reference_map$", key_str)

            if image_match:
                shared_index = f"{image_match.group(1)}_{image_match.group(2)}"
                image_keys[shared_index] = key_str
            elif reference_match:
                shared_index = f"{reference_match.group(1)}_{reference_match.group(2)}"
                reference_keys[shared_index] = key_str

        # Match image keys with reference keys based on shared index
        for shared_index in image_keys.keys() & reference_keys.keys():
            matches.append((image_keys[shared_index], reference_keys[shared_index]))

    return matches



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
    df['reference_map'] = df['patch_id'].apply(lambda x: x + '_reference_map')
    df = df[['s1_name', 'reference_map', 'split']]

    # Split the data into train, validation, and test sets according to the 'split' column
    train = df[df['split'] == 'train']
    validation = df[df['split'] == 'validation']
    test = df[df['split'] == 'test']

    # Create lists of (image_key, reference_key) pairs
    matches_train = []
    matches_validation = []
    matches_test = []

    for index, row in train.iterrows():
        matches_train.append((row['s1_name'], row['reference_map']))

    for index, row in validation.iterrows():
        matches_validation.append((row['s1_name'], row['reference_map']))

    for index, row in test.iterrows():
        matches_test.append((row['s1_name'], row['reference_map']))

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

def keys_from_match(idx, matches):
    """
    Get the image and reference map keys from the matches list based on the index.

    Args:
        idx (int): The index of the match.
        matches (List[Tuple[str, str]]): A list of (image_key, reference_key) pairs.

    Returns:
        Tuple[str, str]: The image key and reference key.
    """
    image_key, reference_key = matches[idx]
    return image_key, reference_key

'''
Mask Functions
'''

def one_hot_to_class_indices(input_mask):
    """
    Converts a one-hot encoded mask to class indices.
    """
    return input_mask.argmax(dim=0)

# Function to replace pixel values with class indices
def replace_pixel_values_with_class_indices(mask, pixel_value_to_class_index):
    """
    Replaces pixel values with class indices in the mask.
    """ 
    class_indices_mask = np.zeros_like(mask, dtype=np.int32)

    # Iterate over the pixel values and class indices
    for value, index in pixel_value_to_class_index.items():
        class_indices_mask[mask == value] = index

    return class_indices_mask

'''
Visualization Functions
'''

def apply_color_map(mask, color_map):
    """
    Apply a color map to a mask.

    Args:
        mask (np.ndarray): The mask to apply the color map to.
        color_map (Dict[int, List[int]]): A dictionary mapping class indices to RGB colors.

    Returns:
        np.ndarray: The mask with the color map applied.
    """
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_index, color in color_map.items():
        mask_rgb[mask == class_index] = color
    return mask_rgb

def visual_from_lmdb_with_ref(reference_string, bands, lmdb_path):
    """
    Show a visualization of the SAR image and its reference map directly from the LMDB database.

    Args:
        reference_string (str): The reference string to visualize.
        bands (List[str]): The bands to visualize.
        lmdb_path (str): Path to the LMDB database.
    """
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        safetensor_dict = load(txn.get(reference_string.encode()))
    
    tensor = np.stack([safetensor_dict[band] for band in bands])
    
    # Plotting
    fig, axes = plt.subplots(1, len(bands), figsize=(18, 6))
    if len(bands) == 1:
        axes = [axes]  # Make axes a list if there's only one subplot

    for i, band in enumerate(bands):
        axes[i].imshow(tensor[i], cmap='gray')
        axes[i].set_title(f'{reference_string} - {band}')
        axes[i].axis('off')

    plt.show()

def display_results(model, loader, lmdb_path, num_images=10, color_map=color_map, pixel_value_to_class_index=pixel_value_to_class_index, indices = None):
    """
    Displays n random masks from an image set with the ground truth and predicted masks.

    Args:
        model (torch.nn.Module): The model to use for inference.
        loader (torch.utils.data.DataLoader): The data loader for the image set.
        lmdb_path (str): Path to the LMDB database.
        num_images (int): The number of images to display.
        color_map (Dict[int, List[int]]): A dictionary mapping class indices to RGB colors.
        pixel_value_to_class_index (Dict[int, int]): A dictionary mapping pixel values to class

    """
    model.eval()
    env = lmdb.open(lmdb_path, readonly=True, lock = False)
    
    # Create a list of all indices in the test set
    all_indices = list(range(len(loader.dataset)))
    
    if indices is None:
    # Randomly sample indices
        indices = random.sample(all_indices, num_images)
    print(indices)
    
    for idx in indices:
        # Get the image and mask at the sampled index
        image, mask = loader.dataset[idx]
        mask = mask.argmax(dim=0).cpu().numpy()

        # Get the corresponding real mask from the LMDB database
        image_key, reference_key = loader.dataset.matches[idx]
        with env.begin() as txn:
            real_mask_data = load(txn.get(reference_key.encode()))
        real_mask = real_mask_data["Data"]

        print(f"Image Key: {image_key}, Reference Key: {reference_key}")

        # Predict the mask
        pred = inference(image.unsqueeze(0).to(device), model)
        
        real_mask_indices = replace_pixel_values_with_class_indices(real_mask, pixel_value_to_class_index)

        real_mask_colored = apply_color_map(real_mask_indices, color_map)
        pred_mask_colored = apply_color_map(pred.squeeze(0).cpu(), color_map)

        # Create a figure with four subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Display the VH channel
        axes[0].imshow(image[0].cpu().numpy(), cmap='gray')
        axes[0].set_title('VH Channel Visualization')
        axes[0].axis('off')

        # Display the VV channel
        axes[1].imshow(image[1].cpu().numpy(), cmap='gray')
        axes[1].set_title('VV Channel Visualization')
        axes[1].axis('off')

        # Display the ground truth mask
        axes[2].imshow(real_mask_colored)
        axes[2].set_title('Ground Truth Mask')
        axes[2].axis('off')

        # Display the predicted mask
        axes[3].imshow(pred_mask_colored)
        axes[3].set_title('Predicted Mask')
        axes[3].axis('off')

        plt.show()

def display_good_bad(model, lmdb_path, ref_image, ref_reference, color_map = color_map, pixel_value_to_class_index = pixel_value_to_class_index):
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
        padded_img_tensor = pad_image(image_tensor, 128, 128)
        
        # Load reference map
        real_mask_data = load(txn.get(ref_reference.encode()))
        real_mask = real_mask_data["Data"]

        # Predict the mask
        pred = inference(padded_img_tensor, model)
        pred_mask_colored = apply_color_map(pred.squeeze(0).cpu(), color_map)

        real_mask_indices = replace_pixel_values_with_class_indices(real_mask, pixel_value_to_class_index)
        real_mask_colored = apply_color_map(real_mask_indices, color_map)

        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Display the VH channel
        axes[0].imshow(image[0], cmap='gray')
        axes[0].set_title('VH Channel Visualization')
        axes[0].axis('off')

        # Display the ground truth mask
        axes[1].imshow(real_mask_colored)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')

        # Display the predicted mask
        axes[2].imshow(pred_mask_colored)
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')

        plt.show()

'''
Model Creation and Loading Utilities
'''
def create_base_model(backbone ='resnet50', weights = None, in_channel = 2, num_classes = 20):
    model = Unet(
        encoder_name= backbone,     # Pretrained encoder, adjust if necessary
        encoder_weights=weights,    # No ImageNet weights since SAR images are different
        in_channels=in_channel,     # Two bands (VH, VV)
        classes=num_classes,        # Number of segmentation classes
        activation="softmax",       # Output activation
    )
    return model

def load_from_checkpoint(checkpoint_path, num_classes= 20):
    model = create_base_model(num_classes = num_classes)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(torch.load(checkpoint_path, weights_only = True), strict=False)
    return model

def load_base_with_bigearth_pretrained(num_classes= 20):
    # Load the pretrained model
    model = create_base_model(num_classes = num_classes)
    model_bigearth_classifier = BigEarthNetv2_0_ImageClassifier.from_pretrained(
        "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.1.1"
    )
    pretrained_weights = model_bigearth_classifier.state_dict()

    # Create a mapping from the pretrained model keys to the untrained model keys
    key_mapping = {
        pretrained_key: pretrained_key.replace("model.vision_encoder", "encoder")
        for pretrained_key in pretrained_weights.keys()
        if pretrained_key.startswith("model.vision_encoder")
    }

    # Map weights to the untrained model
    mapped_state_dict = {
        untrained_key: pretrained_weights[pretrained_key]
        for pretrained_key, untrained_key in key_mapping.items()
    }
    # Load the model
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    return model

'''
120x120 Model
'''

class CustomDecoder(nn.Module):
    def __init__(self, in_channels, decoder_channels):
        super().__init__()

        # Define upsampling blocks with transposed convolution + ConvBlock
        def up_block(in_ch, out_ch, scale_factor):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.up1 = up_block(in_channels, decoder_channels[0], scale_factor=8/4)     # 4 → 8
        self.up2 = up_block(decoder_channels[0], decoder_channels[1], scale_factor=15/8)            # 8 → 15
        self.up3 = up_block(decoder_channels[1], decoder_channels[2], scale_factor=30/15)            # 15 → 30
        self.up4 = up_block(decoder_channels[2], decoder_channels[3], scale_factor=60/30)             # 30 → 60
        self.up5 = up_block(decoder_channels[3], decoder_channels[4], scale_factor=120/60)            # 60 → 120
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x

class CustomUnet(smp.Unet):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights = None,
        decoder_channels = (256, 128, 64, 32, 16),
        in_channels: int = 2,
        classes: int = 20,
        activation = "softmax",
    ):
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, activation=activation)
        
        # Modify the decoder to have the desired upsampling stages
        self.decoder = CustomDecoder(in_channels=self.encoder.out_channels[-1], decoder_channels=decoder_channels)
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]
        x = self.decoder(x)
        x = self.segmentation_head(x)
        return x

def create_base_model120(backbone='resnet50', weights=None, in_channels=2, num_classes=20):
    model = CustomUnet(
        encoder_name=backbone,     # Pretrained encoder, adjust if necessary
        encoder_weights=weights,   # No ImageNet weights since SAR images are different
        in_channels=in_channels,   # Two bands (VH, VV)
        classes=num_classes,       # Number of segmentation classes
    )
    return model

def load_from_checkpoint120(checkpoint_path):
    model = create_base_model120()
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(torch.load(checkpoint_path, weights_only = True), strict=False)
    return model

def load_base_with_bigearth_pretrained120():
    # Load the pretrained model
    model = create_base_model120()
    model_bigearth_classifier = BigEarthNetv2_0_ImageClassifier.from_pretrained(
        "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.1.1"
    )
    pretrained_weights = model_bigearth_classifier.state_dict()

    # Create a mapping from the pretrained model keys to the untrained model keys
    key_mapping = {
        pretrained_key: pretrained_key.replace("model.vision_encoder", "encoder")
        for pretrained_key in pretrained_weights.keys()
        if pretrained_key.startswith("model.vision_encoder")
    }

    # Map weights to the untrained model
    mapped_state_dict = {
        untrained_key: pretrained_weights[pretrained_key]
        for pretrained_key, untrained_key in key_mapping.items()
    }
    # Load the model
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    return model

'''
Training and Inference Utilities
'''

def training(model, epoch_start, epoch_end, loss_fn, train_loader, val_loader, num_classes, model_name = "unet120"):
    print(f"Training {model_name} from epoch {epoch_start} to {epoch_end}")
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    # Iterate over the epochs
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.1, patience=3, verbose=True
    # )
    # Mixed precision training setup
    # scaler = GradScaler()

    for epoch in range(epoch_start, epoch_end+1):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_f1 = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{epoch_end}", unit="batch")
        for images, masks in train_loader_tqdm:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(device_type=device_type):
                outputs = model(images)
                masks = masks.argmax(dim=1)  # Convert one-hot to class indices
                loss = loss_fn(outputs, masks)

            # Backward pass with gradient scaling
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            loss.backward()
            optimizer.step()

            # Compute IoU metrics
            tp, fp, fn, tn = smp.metrics.get_stats(
                outputs.argmax(dim=1).to(torch.int32), masks, mode="multiclass", num_classes=num_classes
            )
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

            train_loss += loss.item()
            train_iou += iou.item()
            train_f1 += f1.item()

            # Clear CUDA cache
            # torch.cuda.empty_cache()


        # Update tqdm progress bar
        epoch_loss_train = train_loss / len(train_loader)
        epoch_iou_train = train_iou / len(train_loader)
        epoch_f1_train = train_f1 / len(train_loader)

        train_loader_tqdm.set_postfix(loss=epoch_loss_train, iou=epoch_iou_train, f1=epoch_f1_train)

        print(f"Epoch {epoch}, Loss: {epoch_loss_train:.4f}, IoU: {epoch_iou_train:.4f}, F1: {epoch_f1_train:.4f}")

        # Log training metrics
        writer.add_scalar('Loss/train', epoch_loss_train, epoch)
        writer.add_scalar('IoU/train', epoch_iou_train, epoch)
        writer.add_scalar('F1/train', epoch_f1_train, epoch)


        # Validation loop
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_f1 = 0.0
        val_loader_tqdm = tqdm(val_loader, desc="Validation", unit="batch")

        with torch.no_grad():
            for images, masks in val_loader_tqdm:
                images, masks = images.to(device), masks.to(device)

                # Mixed precision forward pass
                with autocast(device_type=device_type):
                    outputs = model(images)
                    masks = masks.argmax(dim=1)  # Convert one-hot to class indices
                    loss = loss_fn(outputs, masks)

                tp, fp, fn, tn = smp.metrics.get_stats(
                    outputs.argmax(dim=1).to(torch.int32), masks, mode="multiclass", num_classes=num_classes
                )
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

                val_loss += loss.item()
                val_iou += iou.item()
                val_f1 += f1.item()

                # Clear CUDA cache
                # torch.cuda.empty_cache()

        # Update tqdm progress bar
        epoch_loss_val = val_loss / len(val_loader)
        epoch_iou_val = val_iou / len(val_loader)
        epoch_f1_val = val_f1 / len(val_loader)

        val_loader_tqdm.set_postfix(loss=epoch_loss_val, iou=epoch_iou_val, f1=epoch_f1_val)

        print(f"Epoch {epoch}, Loss: {epoch_loss_val:.4f}, IoU: {epoch_iou_val:.4f}, F1: {epoch_f1_val:.4f}")

        # Log val metrics
        writer.add_scalar('Loss/val', epoch_loss_val, epoch)
        writer.add_scalar('IoU/val', epoch_iou_val, epoch)
        writer.add_scalar('F1/val', epoch_f1_val, epoch)

        # Step the scheduler
        # scheduler.step(val_loss)

        # Save model checkpoints (optional)
        torch.save(model.state_dict(), f"../models/{model_name}_epoch_{epoch}.pth")

    writer.flush()
    writer.close()

def calculate_scores(model, test_loader, device, num_classes):
    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    test_f1 = 0.0
    criterion = nn.CrossEntropyLoss()  # Adjust based on your task

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

    Args:
        img (torch.Tensor): Input image tensor.  shape: [2, height, width]
        model (torch.nn.Module): Trained model.

    Returns:
        torch.Tensor: Predicted mask tensor. shape: [height, width]
    """
    img = img.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1)
    return pred

'''
Dataset and DataLoader Utilities
'''
from torch.utils.data import Dataset
class SARSegmentationDataset(Dataset):
    def __init__(self, lmdb_path, matches, num_classes=20, target_height=128, target_width=128, transform=None):
        self.image_lmdb_file = lmdb_path
        self.env = None
        self.matches = matches
        self.num_classes = num_classes
        self.target_height = target_height
        self.target_width = target_width
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
                max_spare_txns=16,      # expected number of concurrent transactions (e.g. threads/workers)
            )

    def __len__(self):
        return len(self.matches)

    def pad_image(self, img_tensor):
        if img_tensor.ndim == 2:  # If the tensor has only two dimensions (H, W)
            img_tensor = img_tensor.unsqueeze(0)  # Add a channel dimension (1, H, W)
        _, h, w = img_tensor.shape
        pad_h = self.target_height - h
        pad_w = self.target_width - w
        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        padded_img = F.pad(img_tensor, padding, mode='constant', value=0)
        return padded_img

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
        # Replace pixel values with class indices
        mask_indices = replace_pixel_values_with_class_indices(mask_data, pixel_value_to_class_index)

        # Ensure all class indices are within the valid range
        mask_indices = np.clip(mask_indices, 1, self.num_classes-1)

        # One-hot encode the class indices
        mask_indices_one_hot = F.one_hot(torch.tensor(mask_indices).long(), num_classes=self.num_classes).permute(2, 0, 1).float() # (H, W, C) -> (C, H, W)

        # Pad image and mask
        image_tensor = self.pad_image(torch.tensor(image_tensor, dtype=torch.float32))
        mask_indices_one_hot = self.pad_image(mask_indices_one_hot)

        # Convert tensors to numpy arrays for albumentations
        image_tensor_np = image_tensor.numpy()
        mask_indices_one_hot_np = mask_indices_one_hot.numpy()

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image_tensor_np, mask=mask_indices_one_hot_np)
            image_tensor_np = augmented['image']
            mask_indices_one_hot_np = augmented['mask']

        # Convert back to tensors
        image_tensor = torch.tensor(image_tensor_np, dtype=torch.float32)
        mask_indices_one_hot = torch.tensor(mask_indices_one_hot_np, dtype=torch.long)

        return image_tensor, mask_indices_one_hot

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
                max_spare_txns=16,      # expected number of concurrent transactions (e.g. threads/workers)
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
        # Replace pixel values with class indices
        mask_indices = replace_pixel_values_with_class_indices(mask_data, pixel_value_to_class_index)

        # Ensure all class indices are within the valid range
        mask_indices = np.clip(mask_indices, 1, self.num_classes-1)

        # One-hot encode the class indices
        mask_indices_one_hot = F.one_hot(torch.tensor(mask_indices).long(), num_classes=self.num_classes).permute(2, 0, 1).float().numpy() # (H, W, C) -> (C, H, W)

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image_tensor, mask=mask_indices_one_hot)
            image_tensor = augmented['image']
            mask_indices_one_hot = augmented['mask']

        # Convert back to tensors
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
        mask_indices_one_hot = torch.tensor(mask_indices_one_hot, dtype=torch.long)

        return image_tensor, mask_indices_one_hot

'''
Experiment Utilities
'''
def get_encoder_sequential_layer_output_shapes(model, input_tensor):
    """
    Get the output shapes of the encoder's sequential layers.

    Args:
        model (torch.nn.Module): The model.
        input_tensor (torch.Tensor): The input tensor.

    Returns:
        Dict[torch.nn.Module, torch.Size]: A dictionary mapping the layer to its output shape.
    """
    layer_shapes = {}

    def hook(module, input, output):
        layer_shapes[module] = output.shape

    hooks = []
    for name, layer in model.encoder.named_children():
        if isinstance(layer, nn.Sequential):
            hooks.append(layer.register_forward_hook(hook))

    model(input_tensor)

    for hook in hooks:
        hook.remove()

    for i, (layer, shape) in enumerate(layer_shapes.items()):
        print(f"Layer {i + 1}: {shape}")

    return layer_shapes

'''
Activation LMDB Utilities
'''

def load_activation(image_reference, layer_name, env):
    """
    Find and load the activation for a specific image and layer.

    Args:
        image_reference (str): The image reference.
        layer_name (str): The layer name.

    Returns:
        torch.Tensor: The loaded activation tensor.

    """
    with env.begin() as txn:
        # Construct the key to retrieve the activation (same key format used when saving)
        key = f"{image_reference}_{layer_name}".encode()
        
        # Retrieve the activation from the LMDB database
        activation_data = txn.get(key)
        
        if activation_data is not None:
            # Convert the byte data back to a numpy array
            activation_array = np.frombuffer(activation_data, dtype=np.float32)
            
            # Determine the correct shape based on the layer name
            if layer_name == 'encoder.layer1':
                activation_shape = (1, 256, 32, 32)
            elif layer_name == 'encoder.layer2':
                activation_shape = (1, 512, 16, 16)
            elif layer_name == 'encoder.layer3':
                activation_shape = (1, 1024, 8, 8)
            elif layer_name == 'encoder.layer4':
                activation_shape = (1, 2048, 4, 4)
            else:
                raise ValueError(f"Unknown layer name: {layer_name}")
            
            # Reshape the array to match the original activation shape
            activation_array = activation_array.reshape(activation_shape)
            
            # Convert it to a PyTorch tensor if needed
            activation_tensor = torch.tensor(activation_array)
            
            return activation_tensor
        else:
            print(f"Activation for {image_reference} and {layer_name} not found.")
            return None

def extract_region_activations(activation, region_coords, layer_name):
    if layer_name == 'encoder.layer1':
        factor = 4
    elif layer_name == 'encoder.layer2':
        factor = 8
    elif layer_name == 'encoder.layer3':
        factor = 16
    elif layer_name == 'encoder.layer4':
        factor = 32
    else:
        raise ValueError(f"Unknown layer name: {layer_name}")
    
    x1, y1, x2, y2 = region_coords
    new_x1 = x1 // factor
    new_y1 = y1 // factor
    new_x2 = x2 // factor
    new_y2 = y2 // factor

    # print(f"New Region Coords: ({new_x1}, {new_y1}, {new_x2}, {new_y2})")
    region_activation = activation[:, :, new_y1:new_y2, new_x1:new_x2]

    return region_activation

def get_activation(name, dictionary):
    def hook(model, input, output):
        dictionary[name] = output.detach()  # Capture the output and detach from the computation graph
    return hook

def setup_hooks(model, activations_dict):
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Sequential) and name.startswith("encoder") and "downsample" not in name:
            layer.register_forward_hook(get_activation(name, activations_dict))

'''
Similarity Utilities
'''

def compute_similarity(query_activations, train_activations, metric='cosine'):
    query_flat = query_activations.flatten().cpu().numpy()
    train_flat = train_activations.flatten().cpu().numpy()
    return 1 - cdist([query_flat], [train_flat], metric=metric)[0][0]

def plot_similarities(query_image, query_predicted, similar_images, similar_masks, titles):
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
    axes[0, 0].imshow(query_image[1].cpu().numpy(), cmap='gray')  # First Band
    axes[0, 0].set_title("Query Image")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(query_predicted)  # Query mask (HW)
    axes[1, 0].set_title("Predicted Mask")
    axes[1, 0].axis("off")

    # Display Top-5 Similar Images and Masks
    for i, (img, mask, title) in enumerate(zip(similar_images, similar_masks, titles), start=1):
        axes[0, i].imshow(img[0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f"Image {i}: {title}")
        axes[0, i].axis("off")
        axes[1, i].imshow(mask)  # Mask (HW)
        axes[1, i].set_title(f"Mask {i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

def find_n_similar_images(query_activation, layer_name, train_image_keys, activation_lmdb_path, n=10, region_coords=None, metric = 'cosine'):
    min_heap = []
    temp_query_activation = query_activation
    env = lmdb.open(activation_lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        for train_image_key in train_image_keys:
            train_activation = load_activation(train_image_key, layer_name, env = env)

            if train_activation is not None:
                if region_coords is not None:
                    # print("Changing scope of comparison from full image to region. Size before: ", train_activation.shape)
                    train_activation = extract_region_activations(train_activation, region_coords, layer_name)
                    temp_query_activation = extract_region_activations(query_activation, region_coords, layer_name)
                    # print("Train Activation Shape: ", train_activation.shape)
                    # print("Query Activation Shape: ", temp_query_activation.shape)

                similarity = compute_similarity(temp_query_activation, train_activation, metric=metric)
                
                if len(min_heap) < n:
                    heapq.heappush(min_heap, (similarity, train_image_key))
                else:
                    heapq.heappushpop(min_heap, (similarity, train_image_key))

    top_n_results = sorted(min_heap, key=lambda x: x[0], reverse=True)
    env.close()
    return top_n_results

def display_n_similar_images(query_image, activations_dict, layer_name, image_keys, activations_lmdb_path, images_lmdb_path, model, img_hw = (128,128), color_map = color_map, region_coords=None):
    """
    Display the query image and the top N similar images with their predicted masks.

    Args:
        query_image (torch.Tensor): The query image tensor in (C, H, W) format.
        top_n_results (List[Tuple[float, str]]): A list of (similarity_score, image_key) pairs.
        lmdb_path (str): Path to the LMDB database.
        model (torch.nn.Module): The model used for inference.
        img_hw (Tuple[int, int]): The target image height and width.
        color_map (Dict[int, List[int]]): A dictionary mapping class indices to RGB colors.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_hooks(model, activations_dict)
    activations_dict.clear()
    
    # Get the query mask
    query_image = torch.tensor(query_image, dtype=torch.float32).to(device)
    query_image = pad_image(query_image.unsqueeze(0), img_hw[0], img_hw[1])

    model.eval()
    output = inference(query_image, model)
    out_mask_colored = apply_color_map(output.squeeze(0).cpu(), color_map)
    
    query_activations = activations_dict[layer_name]

    # Find similar images based on activations
    similar_images = []
    similar_masks = []
    titles = []
    top_n_results = find_n_similar_images(query_activations, layer_name, image_keys, activations_lmdb_path, n=5, region_coords=region_coords)
    print("Top N Results: ", top_n_results)
    
    for similarity_score, image_key in top_n_results:
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
        titles.append(f"{image_key}\nSim: {similarity_score:.3f}")

    plot_similarities(
        query_image=query_image.squeeze(0),  # Query image (C, H, W)
        query_predicted=out_mask_colored,    # Query mask (H, W)
        similar_images=similar_images,      # Top-5 images
        similar_masks=similar_masks,        # Top-5 masks
        titles=titles                        # Titles with similarity scores
    )


''' 
Multi-Purpose Utilities
'''

def pad_image(img_tensor, target_height, target_width):
    """
    Pads an image tensor to the target height and width.

    Args:
        img_tensor (torch.Tensor): The image tensor to pad.
        target_height (int): The target height.
        target_width (int): The target width.

    Returns:
        torch.Tensor: The padded image tensor.
    """
    _, _, h, w = img_tensor.shape
    pad_h = target_height - h
    pad_w = target_width - w
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    padded_img = F.pad(img_tensor, padding, mode='constant', value=0)
    return padded_img

