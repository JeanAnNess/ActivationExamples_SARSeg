import torch
import numpy as np
from ..config import SCALE_FACTORS
from ..models.registry import get_layer_config


def get_activation(name, dictionary):
    def hook(model, input, output):
        dictionary[name] = output.detach()  # Capture the output and detach from the computation graph
    return hook


def setup_hooks(model, activations_dict, hook_filter=None):
    """
    Set up hooks to capture activations from specific layers of the model.

    Note: Specific for the current implementation of the model.
    """
    if hook_filter is None:
        def hook_filter(name, layer):
            return isinstance(layer, torch.nn.Sequential) and (name.startswith("encoder") or name.startswith("decoder")) and "downsample" not in name

    for name, layer in model.named_modules():
        if hook_filter(name, layer):
            layer.register_forward_hook(get_activation(name, activations_dict))


def extract_region_activations(activation, region_coords, layer_name, scale_factors=None):
    """
    Extracts a specific region from the activation tensor based on the provided coordinates and layer name.
    """
    if scale_factors is None:
        scale_factors = SCALE_FACTORS

    factor = scale_factors.get(layer_name)
    if factor is None:
        raise ValueError(f"Layer {layer_name} not found in scale_factors.")

    x1, y1, x2, y2 = region_coords
    # Scale coordinates based on the factor
    scaled_x1 = int(np.round(x1 / factor))
    scaled_y1 = int(np.round(y1 / factor))
    scaled_x2 = int(np.ceil(x2 / factor))
    scaled_y2 = int(np.ceil(y2 / factor))

    extracted = activation[:, :, scaled_y1:scaled_y2, scaled_x1:scaled_x2]
    return extracted


def load_activation(image_reference, layer_name, env, arch_name="unet", dtype=torch.float32):
    """
    Load the activation tensor from LMDB.
    """
    layer_shapes, _ = get_layer_config(arch_name)
    with env.begin() as txn:
        key = f"{image_reference}_{layer_name}".encode()
        activation_data = txn.get(key)
        if activation_data is None:
            return None
        import numpy as np
        read_dtype = np.float16 if dtype == torch.float16 else np.float32
        activation_array = np.frombuffer(activation_data, dtype=read_dtype)
        activation_tensor = torch.tensor(activation_array.copy(), dtype=dtype).reshape(layer_shapes[layer_name])
        return activation_tensor
