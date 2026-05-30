import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

color_map = {
    1:  [206, 206, 206],
    2:  [64,  64,  64],
    3:  [254, 215, 119],
    4:  [254, 187, 71],
    5:  [253, 152, 40],
    6:  [239, 120, 24],
    7:  [216, 89,  8],
    8:  [183, 66,  2],
    9:  [200, 232, 154],
    10: [162, 216, 137],
    11: [119, 197, 120],
    12: [75,  176, 98],
    13: [46,  146, 76],
    14: [21,  120, 62],
    15: [210, 180, 140],
    16: [182, 212, 233],
    17: [120, 181, 216],
    18: [63,  143, 196],
    19: [23,  100, 171],
    20: [255, 255, 255],
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


# Function to replace pixel values with class indices
def replace_pixel_values_with_class_indices(mask, mapping=None):
    """
    Replaces pixel values with class indices in the mask.
    """
    if mapping is None:
        mapping = pixel_value_to_class_index
    class_indices_mask = np.zeros_like(mask, dtype=np.int32)
    for value, index in mapping.items():
        class_indices_mask[mask == value] = index
    return class_indices_mask


def apply_color_map(mask, cmap=None):
    """
    Apply a color map to a mask.
    """
    if cmap is None:
        cmap = color_map
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_index, color in cmap.items():
        mask_rgb[mask == class_index] = color
    return mask_rgb
