import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from tqdm import tqdm
from ..config import device


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
             save_dir="../models/",
             ignore_index=20,
             input_size=None,
             optimizer=None,
             scheduler=None,
             best_val_loss=None,
             patience_counter=0):
    """
    Training loop with combined Focal + Dice loss, per-class IoU/F1 logging, and tqdm progress.
    
    For incremental training (e.g. epochs 1-3 then 4-5 then 6-10):
      1. First run: pass only epoch_start/epoch_end
      2. After each run, a full checkpoint is saved
      3. Resume: load_training_checkpoint() then pass optimizer=, scheduler=, best_val_loss=, patience_counter=
    """
    writer = SummaryWriter()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    focal_loss = FocalLoss(mode="multiclass", ignore_index=ignore_index)
    dice_loss = DiceLoss(mode="multiclass", ignore_index=ignore_index)
    w_focal, w_dice = loss_weights

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) if optimizer is None else optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2) if scheduler is None else scheduler

    scaler = GradScaler()  # Mixed precision
    best_val_loss = float("inf") if best_val_loss is None else best_val_loss
    patience = 5  # Early stopping patience
    patience_counter = 0 if patience_counter is None else patience_counter

    for epoch in range(epoch_start, epoch_end + 1):
        if epoch == freeze_epochs + 1:
            for param in model.encoder.parameters():
                param.requires_grad = True
            print(f"Unfroze encoder at epoch {epoch}")

        model.train()
        train_loss, train_iou, train_f1 = 0.0, 0.0, 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{epoch_end}", unit="batch")
        num_batches = 0

        for images, masks in train_loader_tqdm:
            images, masks = images.to(device), masks.to(device)
            masks = masks.argmax(dim=1)  # Convert one-hot to class indices

            if input_size is not None:
                images = F.pad(images, (0, input_size[1] - images.shape[3], 0, input_size[0] - images.shape[2]), mode="constant", value=0)
                mask_pad = F.pad(masks.unsqueeze(1).float(), (0, input_size[1] - masks.shape[2], 0, input_size[0] - masks.shape[1]), mode="constant", value=ignore_index)
                masks_to_use = mask_pad.squeeze(1).long()
            else:
                masks_to_use = masks

            optimizer.zero_grad()
            with autocast(device_type=device_type):
                outputs = model(images)
                loss_f = focal_loss(outputs, masks_to_use)
                loss_d = dice_loss(outputs, masks_to_use)
                loss = w_focal * loss_f + w_dice * loss_d

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()

            tp, fp, fn, tn = smp.metrics.get_stats(
                outputs.argmax(dim=1).to(torch.int32), masks_to_use, mode="multiclass", num_classes=num_classes
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

        epoch_loss_train = train_loss / len(train_loader)
        epoch_iou_train = train_iou / len(train_loader)
        epoch_f1_train = train_f1 / len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss_train, epoch)
        writer.add_scalar("IoU/train", epoch_iou_train, epoch)
        writer.add_scalar("F1/train", epoch_f1_train, epoch)
        print(f"Epoch {epoch}, Train Loss: {epoch_loss_train:.4f}, IoU: {epoch_iou_train:.4f}, F1: {epoch_f1_train:.4f}")

        model.eval()
        val_loss, val_iou, val_f1 = 0.0, 0.0, 0.0
        val_loader_tqdm = tqdm(val_loader, desc="Validation", unit="batch")
        num_batches = 0

        with torch.no_grad():
            for images, masks in val_loader_tqdm:
                images, masks = images.to(device), masks.to(device)
                masks = masks.argmax(dim=1)

                if input_size is not None:
                    images = F.pad(images, (0, input_size[1] - images.shape[3], 0, input_size[0] - images.shape[2]), mode="constant", value=0)
                    mask_pad = F.pad(masks.unsqueeze(1).float(), (0, input_size[1] - masks.shape[2], 0, input_size[0] - masks.shape[1]), mode="constant", value=ignore_index)
                    masks_to_use = mask_pad.squeeze(1).long()
                else:
                    masks_to_use = masks

                with autocast(device_type=device_type):
                    outputs = model(images)
                    loss_f = focal_loss(outputs, masks_to_use)
                    loss_d = dice_loss(outputs, masks_to_use)
                loss = w_focal * loss_f + w_dice * loss_d

                tp, fp, fn, tn = smp.metrics.get_stats(
                    outputs.argmax(dim=1).to(torch.int32), masks_to_use, mode="multiclass", num_classes=num_classes
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
            _save_training_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, patience_counter, epoch_loss_val, f"{save_dir}/{model_name}_best_checkpoint.pth")
        else:
            patience_counter += 1

        _save_training_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, patience_counter, epoch_loss_val, f"{save_dir}/{model_name}_checkpoint_epoch_{epoch}.pth")
        
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    last_lr = optimizer.param_groups[0]["lr"]
    print(f"Last used learning rate: {last_lr}")

    writer.flush()
    writer.close()
    return model, optimizer


def calculate_scores(model, test_loader, device, num_classes, ignore_index=20, input_size=None):
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
            # Convert one-hot encoded masks to class indices
            masks = masks.argmax(dim=1)

            if input_size is not None:
                images = F.pad(images, (0, input_size[1] - images.shape[3], 0, input_size[0] - images.shape[2]), mode="constant", value=0)
                mask_pad = F.pad(masks.unsqueeze(1).float(), (0, input_size[1] - masks.shape[2], 0, input_size[0] - masks.shape[1]), mode="constant", value=ignore_index)
                masks_to_use = mask_pad.squeeze(1).long()
            else:
                masks_to_use = masks

            outputs = model(images)
            loss = criterion(outputs, masks_to_use)
            tp, fp, fn, tn = smp.metrics.get_stats(
                outputs.argmax(dim=1).to(torch.int32), masks_to_use, mode="multiclass", num_classes=num_classes
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


def inference(img, model, input_size=None):
    """
    Perform inference on a single image.
    """
    # if image not type torch tensor
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    img = img.to(device)
    model = model.to(device)
    model.eval()

    if input_size is not None:
        _, _, h, w = img.shape
        img = F.pad(img, (0, input_size[1] - w, 0, input_size[0] - h), mode="constant", value=0)

    with torch.no_grad():
        output = model(img)
        if input_size is not None:
            pred = output.argmax(dim=1)[:, :h, :w]
        else:
            pred = output.argmax(dim=1)
    return pred


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


def _save_training_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, patience_counter, current_val_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "current_val_loss": current_val_loss,
    }, path)


def load_training_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Load a full training checkpoint including optimizer and scheduler state.
    
    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): If provided, loads optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): If provided, loads scheduler state.
    
    Returns:
        dict: Dictionary with keys 'epoch', 'best_val_loss', 'patience_counter', 'current_val_loss'.
              Also loads model (and optionally optimizer/scheduler) in-place.
    """
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return {
        "epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "patience_counter": checkpoint["patience_counter"],
        "current_val_loss": checkpoint["current_val_loss"],
    }
