import lmdb
import numpy as np
import torch
from safetensors.numpy import load
from tqdm import tqdm
from ..config import device
from ..data.lmdb import get_image_and_mask_from_key
from .hooks import setup_hooks, load_activation


def store_activations(matches, model, lmdb_path, source_lmdb_path, layer_names, break_flag, map_size=2*10**10, hook_filter=None, store_dtype=np.float32):
    """
    Store activations from the model into an LMDB database.
    """
    print("Storing activations in LMDB at path:", lmdb_path)

    # Create LMDB env
    env = lmdb.open(lmdb_path, map_size=map_size)
    source_env = lmdb.open(source_lmdb_path, readonly=True)

    activations = {}
    setup_hooks(model, activations, hook_filter=hook_filter)

    # Iterate over matches and store activations    
    for idx, match in tqdm(enumerate(matches), total=len(matches)):
        if break_flag:
            break
        image_key, reference_key = match

        with source_env.begin() as txn:
            image_data = load(txn.get(image_key.encode()))
            image_bands = ["VH", "VV"]  # Sentinel 1 bands
            image_tensor = np.stack([image_data[band] for band in image_bands])
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0).to(device)

        # Forward, capture activations
        activations.clear()
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # Store activations in LMDB
        batch_activations = {k: v.cpu().numpy().astype(store_dtype) for k, v in activations.items()}
        with env.begin(write=True) as txn:
            for layer_name, activation in batch_activations.items():
                if layer_name not in layer_names:
                    continue
                # Create a unique key for each layer"s activation   
                layer_key = f"{image_key}_{layer_name}"
                txn.put(layer_key.encode(), activation.tobytes())

    env.close()
    source_env.close()


def test_store_activations(model, lmdb_path, source_lmdb, image_reference, layer_name, arch_name="unet", hook_filter=None, dtype=torch.float32):
    """
    Verify that activations are stored correctly in LMDB.
    """
    activations = {}
    setup_hooks(model, activations, hook_filter=hook_filter)

    env = lmdb.open(lmdb_path, readonly=True)
    activation_tensor = load_activation(image_reference, layer_name, env=env, arch_name=arch_name, dtype=dtype)

    if activation_tensor is not None:
        print("Loaded activation shape:", activation_tensor.shape)
    else:
        print("Activation not found")
    env.close()

    image, mask = get_image_and_mask_from_key(image_reference, lmdb_path=source_lmdb)
    image = torch.tensor(image, dtype=torch.float32).to(device).unsqueeze(0)

    # Forward, capture activation
    output = model(image)
    layer_activations = activations[layer_name]

    # Compare stored activation with computed activation
    if torch.allclose(layer_activations.float(), activation_tensor.float().to(device), atol=1e-4, rtol=1e-3):
        print("Activations match successfully!")
