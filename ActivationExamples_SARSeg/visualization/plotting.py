import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import torch
import numpy as np
from ..config import device, color_map as _color_map, apply_color_map, replace_pixel_values_with_class_indices, LAYER_SHAPES
from ..data.lmdb import get_image_and_mask_from_key, get_reference_ref
from ..training.trainer import inference, pad_image


def display_from_image_and_mask(image, mask, color_map=None):
    """
    Displays a visualization of the image and mask.

    Args:
        image (np.ndarray): The image to display.
        mask (np.ndarray): The mask to display. Already in class indices.
        color_map (Dict[int, List[int]]): A dictionary mapping class indices to RGB colors.
    """
    if color_map is None:
        color_map = _color_map
    mask_colored = apply_color_map(mask, color_map)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image[0], cmap="gray")
    axes[0].set_title("VH Channel Visualization")
    axes[0].axis("off")
    axes[1].imshow(mask_colored)
    axes[1].set_title("Mask")
    axes[1].axis("off")
    plt.show()


def display_image_reference_inference(model, lmdb_path, ref_image, ref_reference,
                                       color_map=_color_map,
                                       pixel_value_to_class_index=None, img_size=120):
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
    if pixel_value_to_class_index is None:
        from ..config import pixel_value_to_class_index as _pix
        pixel_value_to_class_index = _pix

    import lmdb as _lmdb
    from safetensors.numpy import load as _load

    model.eval()
    env = _lmdb.open(lmdb_path, readonly=True)

    with env.begin() as txn:
        image_data = _load(txn.get(ref_image.encode()))
        image = np.stack([image_data[band] for band in ["VH", "VV"]])
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

        if img_size == 128:
            padded_img_tensor = pad_image(image_tensor, 128, 128)
        else:
            padded_img_tensor = pad_image(image_tensor, 120, 120)

        # Load reference map
        real_mask_data = _load(txn.get(ref_reference.encode()))
        real_mask = real_mask_data["Data"]

        pred = inference(padded_img_tensor, model)
        pred_mask_colored = apply_color_map(pred.squeeze(0).cpu(), color_map)

        real_mask_indices = replace_pixel_values_with_class_indices(real_mask, pixel_value_to_class_index)
        real_mask_colored = apply_color_map(real_mask_indices, color_map)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(image[0], cmap="gray")
        axes[0].set_title(f"$\\bf{{VH\\ Channel}}$", fontsize=20)
        axes[0].axis("off")
        axes[1].imshow(real_mask_colored)
        axes[1].set_title(f"$\\bf{{Original\\ Reference\\ Map}}$", fontsize=20)
        axes[1].axis("off")
        axes[2].imshow(pred_mask_colored)
        axes[2].set_title(f"$\\bf{{Predicted\\ Mask}}$", fontsize=20)
        axes[2].axis("off")
        plt.show()


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
    target="infer",
    filename=None,
    format="pdf",
    result_name="Example",
    color_map=None,
):
    if color_map is None:
        color_map = _color_map

    x1, y1, x2, y2 = map(int, region)

    img_grayscale_query, reference_map_query = get_image_and_mask_from_key(
        image_ref, mask_ref, lmdb_path=dataset_lmdb_path
    )
    reference_map_query = replace_pixel_values_with_class_indices(reference_map_query)
    reference_map_query_colored = apply_color_map(reference_map_query)
    unique_classes_ref = np.unique(reference_map_query[y1:y2, x1:x2]).tolist()

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

    pos_proto = grid[0, 2].get_position(fig)
    proto_w = pos_proto.width
    proto_h = pos_proto.height

    vertical_gap = 0.05
    block_height = 3 * proto_h + 2 * vertical_gap
    block_y0 = 0.5 - (block_height / 2)

    pos_left = grid[0, 0].get_position(fig)
    left_x = pos_left.x0 + (pos_left.width - proto_w) / 2

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

    prototype_axes = np.empty((total_grid_rows, max_matches), dtype=object)
    for row_idx in range(total_grid_rows):
        for col_offset in range(max_matches):
            ax = fig.add_subplot(grid[row_idx, 2 + col_offset])
            ax.axis("off")
            prototype_axes[row_idx, col_offset] = ax

    for layer_idx, layer_name in enumerate(layers_to_plot):
        img_row = layer_idx * 2
        mask_row = layer_idx * 2 + 1

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

        filtered = df_top_matches[
            (df_top_matches["image_id"] == image_ref) &
            (df_top_matches["original_region"] == tuple(region)) &
            (df_top_matches["layer"] == layer_name)
        ]
        # print(f"Layer: {layer_name}, Matches found: {filtered}")
        matches_raw = ast.literal_eval(str(filtered["top_n"].values[0]))
        matches = matches_raw[:top_x] if isinstance(matches_raw, list) else []

        for proto_col_offset, match in enumerate(matches):
            match_score, match_ref, match_region_raw = match
            match_region = tuple(map(int, match_region_raw))
            px1, py1, px2, py2 = match_region

            prototype_ref = get_reference_ref(match_ref, parquet_path)
            img_proto, proto_refmap = get_image_and_mask_from_key(
                match_ref, prototype_ref, lmdb_path=dataset_lmdb_path
            )
            proto_refmap = replace_pixel_values_with_class_indices(proto_refmap)
            proto_refmap_colored = apply_color_map(proto_refmap)

            mask_infer_proto = inference(img_proto, model).squeeze(0).cpu().numpy()
            proto_mask_colored = apply_color_map(mask_infer_proto)

            if target == "infer":
                plot_map = proto_mask_colored
                unique_proto = np.unique(mask_infer_proto[py1:py2, px1:px2]).tolist()
                title_label = "Inferred\\ Mask"
            else:
                plot_map = proto_refmap_colored
                unique_proto = np.unique(proto_refmap[py1:py2, px1:px2]).tolist()
                title_label = "Reference\\ Map"
            all_unique_labels.update(unique_proto)

            proto_img_full = img_proto[0]

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
            px1, py1, px2, py2 = match_region

            prototype_ref = get_reference_ref(match_ref, parquet_path)
            img_proto, proto_refmap = get_image_and_mask_from_key(
                match_ref, prototype_ref, lmdb_path=dataset_lmdb_path
            )
            proto_refmap = replace_pixel_values_with_class_indices(proto_refmap)
            proto_refmap_colored = apply_color_map(proto_refmap)

            mask_infer_proto = inference(img_proto, model).squeeze(0).cpu().numpy()
            proto_mask_colored = apply_color_map(mask_infer_proto)

            if target == "infer":
                plot_map = proto_mask_colored
                title_label = "Inferred\\ Mask"
            else:
                plot_map = proto_refmap_colored
                title_label = "Reference\\ Map"

            proto_img_full = img_proto[0]

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
        fontsize=16,
        frameon=False,
    )

    grid.update(wspace=0.1, hspace=0.25)
    # plt.tight_layout()
    plt.show()

    if filename:
        fig.savefig(f"{filename}.{format}", format=format, bbox_inches='tight')
        print(f"Figure saved as {filename}")


def show_overlap_matrix(data, targets, layer_names, mode="overlap", title_in=None,
                        save_plot=False, result_name="Example"):
    """
    Visualize the overlap matrix for different layers and targets.

    Args:
        data (Dict): Dictionary containing the overlap data.
        targets (List[str]): List of target names (e.g. "top 5")
        layer_names (List[str]): List of layer names (e.g. "encoder.layer3")
    """
    if not isinstance(layer_names, list):
        layer_names = [layer_names]
    if not isinstance(targets, list):
        targets = [targets]

    # Loop over all targets
    for target in targets:
        matrix = np.zeros((len(layer_names), len(layer_names)))
        for i in range(len(layer_names)):
            for j in range(len(layer_names)):
                if i != j:
                    # Get overlap count for target
                    if (i, j) in data.keys():
                        matrix[i, j] = data[(i, j)][target]
                    elif (j, i) in data.keys():
                        matrix[i, j] = data[(j, i)][target]
                    else:
                        matrix[i, j] = 0  # when i = j

        # Convert matrix to DataFrame
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.colorbar(ax.matshow(matrix, cmap="Blues"))

        # Annotate heatmap
        for i in range(len(layer_names)):
            for j in range(len(layer_names)):
                ax.text(j, i, round(matrix[i, j], 2), ha="center", va="center", color="black")

        if title_in is None:
            if mode == "overlap":
                print(f"Overlap Matrix - {target}")
                title = f"{result_name} Overlap Matrix - {target}"
            else:
                (p1, p2) = data[(0, 1)][f"{result_name} Overlap Top X with Top Y"]
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
            import os
            save_path = f"plots/overlaps_{target}"
            fig.savefig(save_path, bbox_inches="tight")

        plt.show()
