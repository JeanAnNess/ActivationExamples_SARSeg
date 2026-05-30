import numpy as np


def get_top_subset(layer, top_n):
    """
    Return a set of unique keys for the first top_n entries of the layer.
    """
    return {(name, region) for (score, name, region) in layer[:top_n]}


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
    return iou >= eta


def get_pairwise_overlap(list_of_lists, layer_names, eta=0.85):
    """
    Compute pairwise overlaps between Top-{5,10,20,50} elements of each layer in the list_of_lists.
    """
    results = {}
    n = len(layer_names)
    top_5_sets = [get_top_subset(layer, 5) for layer in list_of_lists]
    top_10_sets = [get_top_subset(layer, 10) for layer in list_of_lists]
    top_20_sets = [get_top_subset(layer, 20) for layer in list_of_lists]
    top_50_sets = [get_top_subset(layer, 50) for layer in list_of_lists]

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
                "overlap top 50": overlap_50,
            }
    return results


def get_aggregate_overlaps(all_image_layers, layer_names, eta=0.85):
    """
    Aggregate pairwise overlaps of Top-{5,10,20,50} across all image layers.
    
    Args:
        overlap_results (dict): Dictionary with pairwise overlaps.

    Returns a dictionary with aggregated results.
    """
    aggregated = {}
    for image_layers in all_image_layers:
        results = get_pairwise_overlap(image_layers, layer_names, eta)
        for key, value in results.items():
            if key not in aggregated:
                aggregated[key] = {
                    "First Layer": value["First Layer"],
                    "Second Layer": value["Second Layer"],
                    "top 5": [],
                    "top 10": [],
                    "top 20": [],
                    "top 50": [],
                }
            for metric in ["top 5", "top 10", "top 20", "top 50"]:
                aggregated[key][metric].append(value[metric])

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
