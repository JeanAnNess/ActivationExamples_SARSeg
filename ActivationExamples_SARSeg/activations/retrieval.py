import torch
import torch.nn.functional as F
import numpy as np
import lmdb
import heapq
from ..config import device, LAYER_SHAPES, SCALE_FACTORS, apply_color_map
from ..data.lmdb import get_image_and_mask_from_key
from ..training.trainer import inference, pad_image
from .hooks import setup_hooks, load_activation, extract_region_activations


def compute_similarity(query_activation_flat, candidate_activation_flat, metric="cosine"):
    """
    Compute similarity efficiently using pre-flattened tensors.

    Note: Currently supports "cosine" and "euclidean" metrics.
    """
    if metric == "cosine":
        return F.cosine_similarity(query_activation_flat, candidate_activation_flat, dim=1).item()
    else:
        return -torch.norm(query_activation_flat - candidate_activation_flat, dim=1).item()


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
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    num_images = len(similar_images) + 1  # Query image + similar images
    fig, axes = plt.subplots(2, num_images, figsize=(18, 6))

    # Display Query Image and Mask
    axes[0, 0].imshow(query_image[1].cpu().numpy(), cmap="gray")  # First Band
    axes[0, 0].set_title("Query Image")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(query_predicted)  # Query mask (HW)
    axes[1, 0].set_title("Predicted Mask")
    axes[1, 0].axis("off")

    x1, y1, x2, y2 = draw_boxes.pop(0)
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

        if draw_box:
            x1, y1, x2, y2 = draw_box
            rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
            rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
            axes[0, i].add_patch(rect1)
            axes[1, i].add_patch(rect2)

    plt.tight_layout()
    plt.show()


def find_n_similar_regions(query_activation, layer_name, train_image_keys, activation_lmdb_path,
                           n=5, region_coords=None, metric="cosine", scale_factors=None, arch_name="unet",
                           dtype=torch.float32, candidate_regions=None):
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

    if scale_factors is None:
        scale_factors = SCALE_FACTORS

    # Extract query activation for the specified region
    query_tensor = query_activation.to(device)

    if region_coords:
        query_tensor = extract_region_activations(query_tensor, region_coords, layer_name,
                                                    scale_factors=scale_factors)

    query_tensor_flat = query_tensor.flatten(start_dim=1)
    win_w, win_h = query_tensor.shape[-2:]

    factor = scale_factors[layer_name]

    if candidate_regions is not None:
        candidate_list = candidate_regions
    else:
        acti_w, acti_h = query_activation.shape[-2:]

        divisions_w = int(np.ceil(acti_w / win_w))
        divisions_h = int(np.ceil(acti_h / win_h))

        if divisions_w == 0 or divisions_h == 0:
            print("Error: Query activation map is not square.")
            print(f"Activation Map Dimensions: {acti_w}x{acti_h}, Window Dimensions: {win_w}x{win_h}")
            print(f"temp_query shape: {query_tensor.shape}")
            print(f"query shape: {query_activation.shape}")
            print(f"region_coords: {region_coords}")
            return []

        candidate_list = None

    with env.begin() as txn:
        for train_image_key in train_image_keys:
            train_activation = load_activation(train_image_key, layer_name, env=env, arch_name=arch_name, dtype=dtype)
            if train_activation is None:
                continue

            train_tensor = train_activation.to(device)
            _, _, H, W = train_tensor.shape

            if candidate_list is not None:
                candidates = candidate_list
            else:
                x_positions = list(range(0, W, win_w))
                if divisions_w * win_w > W:
                    x_positions[-1] = W - win_w
                y_positions = list(range(0, H, win_h))
                if divisions_h * win_h > H:
                    y_positions[-1] = H - win_h
                candidates = [(x*factor, y*factor, (x+win_w)*factor, (y+win_h)*factor)
                              for y in y_positions for x in x_positions]

            for (cx1, cy1, cx2, cy2) in candidates:
                # Extract at same spatial size as query, clamped to feature map bounds
                scx = int(np.round(cx1 / factor))
                scy = int(np.round(cy1 / factor))
                if scx + win_w > W:
                    scx = W - win_w
                if scy + win_h > H:
                    scy = H - win_h
                candidate_activation = train_tensor[:, :, scy:scy+win_h, scx:scx+win_w]
                candidate_activation_flat = candidate_activation.flatten(start_dim=1)
                similarity = compute_similarity(query_tensor_flat, candidate_activation_flat, metric=metric)

                region = (cx1, cy1, cx2, cy2)
                heapq.heappush(min_heap, (similarity, train_image_key, region))
                if len(min_heap) > n:
                    heapq.heappop(min_heap)

    env.close()
    return sorted(min_heap, key=lambda x: x[0], reverse=True)  # Sort by highest similarity


def find_n_similar_images(query_image, layer_names, image_keys,
                          activations_lmdb_path, images_lmdb_path, model,
                          img_hw=(120, 120), color_map=None, region_coords=None, n=5,
                          plotting=True, scale_factors=None, layer_shapes=None, arch_name="unet",
                          dtype=torch.float32, candidate_regions=None):
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
    if color_map is None:
        from ..config import color_map as _cmap
        color_map = _cmap
    if scale_factors is None:
        scale_factors = SCALE_FACTORS
    if layer_shapes is None:
        layer_shapes = LAYER_SHAPES

    activations = {}
    setup_hooks(model, activations)
    activations.clear()

    if not isinstance(layer_names, list):
        layer_names = [layer_names]

    # Get the query mask
    query_image = torch.tensor(query_image, dtype=torch.float32).to(device)
    query_image = pad_image(query_image.unsqueeze(0), img_hw[0], img_hw[1])

    model.eval()
    output = inference(query_image, model)
    out_mask_colored = apply_color_map(output.squeeze(0).cpu(), color_map)

    query_activations = [activations[layer_name] for layer_name in layer_names]

    # Find similar images based on activations
    top_n_results = [find_n_similar_regions(qa, layer_name, image_keys, activations_lmdb_path,
                                             n=50, region_coords=region_coords,
                                             scale_factors=scale_factors,
                                             arch_name=arch_name,
                                             dtype=dtype,
                                             candidate_regions=candidate_regions)
                     for layer_name, qa in zip(layer_names, query_activations)]

    if not plotting:
        return top_n_results

    for results, layer_name in zip(top_n_results, layer_names):
        print(f"   Results for layer: {layer_name}")
        print(f"   Top N Results: {results}")
        similar_images, similar_masks, titles, draw_boxes = [], [], [], []
        draw_boxes.append(region_coords)  # Draw the query region as a box

        for similarity_score, image_key, region_descriptor in results[:n]:
            image, _ = get_image_and_mask_from_key(image_key, lmdb_path=images_lmdb_path)
            image = torch.tensor(image, dtype=torch.float32).to(device)
            image = pad_image(image.unsqueeze(0), img_hw[0], img_hw[1])
            # Get the predicted mask
            pred_mask = model(image).argmax(dim=1).squeeze(0).cpu()
            pred_mask = apply_color_map(pred_mask, color_map)

            # Append results
            similar_images.append(image.squeeze(0))  # Remove batch dimension
            similar_masks.append(pred_mask)
            titles.append(f"Sim: {similarity_score:.3f}\nRegion: {region_descriptor}")

            draw_boxes.append(region_descriptor)

        plot_similarities(
            query_image=query_image.squeeze(0),
            query_predicted=out_mask_colored,
            similar_images=similar_images,
            similar_masks=similar_masks,
            titles=titles,
            draw_boxes=draw_boxes,
        )
    return top_n_results
