from ..config import LAYER_SHAPES, SCALE_FACTORS

LAYER_SHAPES_UNET = LAYER_SHAPES
SCALE_FACTORS_UNET = SCALE_FACTORS

LAYER_SHAPES_DEEPLABV3P = {
    "encoder.layer3": (1, 1024, 15, 15),
    "encoder.layer4": (1, 2048, 15, 15),
    "decoder.aspp": (1, 256, 15, 15),
    "decoder.block2": (1, 256, 30, 30),
}

SCALE_FACTORS_DEEPLABV3P = {
    "encoder.layer3": 8,
    "encoder.layer4": 8,
    "decoder.aspp": 8,
    "decoder.block2": 4,
}

ARCH_CONFIGS = {
    "unet": {
        "layer_shapes": LAYER_SHAPES_UNET,
        "scale_factors": SCALE_FACTORS_UNET,
    },
    "deeplabv3p": {
        "layer_shapes": LAYER_SHAPES_DEEPLABV3P,
        "scale_factors": SCALE_FACTORS_DEEPLABV3P,
    },
}


def get_layer_config(arch_name):
    config = ARCH_CONFIGS.get(arch_name)
    if config is None:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(ARCH_CONFIGS.keys())}")
    return config["layer_shapes"], config["scale_factors"]
