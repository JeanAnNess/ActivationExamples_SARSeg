import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from ..config import device
from ..models.registry import LAYER_SHAPES_DEEPLABV3P, SCALE_FACTORS_DEEPLABV3P


def create_base_model_deeplabv3plus(backbone="resnet50", weights=None, in_channel=2, num_classes=20, encoder_output_stride=8):
    model = smp.DeepLabV3Plus(
        encoder_name=backbone,
        encoder_weights=weights,
        in_channels=in_channel,
        classes=num_classes,
        encoder_output_stride=encoder_output_stride,
        activation="softmax",
    )
    return model


def load_from_checkpoint_deeplabv3plus(checkpoint_path, num_classes=20):
    model = create_base_model_deeplabv3plus(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
    return model


def load_base_with_bigearth_pretrained_deeplabv3plus(num_classes=20):
    model = create_base_model_deeplabv3plus(num_classes=num_classes)
    model_bigearth_classifier = BigEarthNetv2_0_ImageClassifier.from_pretrained(
        "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.1.1"
    )
    pretrained_weights = model_bigearth_classifier.state_dict()

    key_mapping = {
        pretrained_key: pretrained_key.replace("model.vision_encoder", "encoder")
        for pretrained_key in pretrained_weights.keys()
        if pretrained_key.startswith("model.vision_encoder")
    }

    mapped_state_dict = {
        untrained_key: pretrained_weights[pretrained_key]
        for pretrained_key, untrained_key in key_mapping.items()
    }
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    return model


DEEPLABV3P_ACTIVATION_LAYERS = [
    "encoder.layer3",
    "encoder.layer4",
    "decoder.aspp",
    "decoder.block2",
]


def _deeplabv3p_hook_filter(name, layer):
    target_prefixes = ("encoder.", "decoder.")
    skip_names = {"downsample"}

    if not isinstance(layer, nn.Sequential):
        return False
    if not name.startswith(target_prefixes):
        return False
    parts = name.split(".")
    if any(s in parts for s in skip_names):
        return False
    return True


DEEPLABV3P_HOOK_FILTER = _deeplabv3p_hook_filter


def setup_hooks_deeplabv3plus(model, activations_dict):
    for name, layer in model.named_modules():
        if _deeplabv3p_hook_filter(name, layer):
            hook = _get_activation(name, activations_dict)
            layer.register_forward_hook(hook)


def _get_activation(name, dictionary):
    def hook(model, input, output):
        dictionary[name] = output.detach()
    return hook
