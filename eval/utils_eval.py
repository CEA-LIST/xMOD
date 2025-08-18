import os
import torch
import numpy as np
import torch.nn as nn
from models import SlotAttentionAutoEncoder
from dataloaders import KITTI3DEvalDataset



def load_model(ckpt_path, resolution, num_slots, hid_dim, input_channels, device):
    """
    Loads a pre-trained model.
    """
    model = SlotAttentionAutoEncoder(
        tuple(resolution), num_slots, hid_dim, input_channels
    ).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    print(f"Model loaded from {ckpt_path}")
    return model


def get_data_loader(data_path, batch_size, num_workers):
    """
    Returns a data loader for the specified dataset.
    """
    test_set = KITTI3DEvalDataset(split="test", root=data_path)
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


def save_predictions(pred, sample_path):
    """
    Saves predictions to a compressed numpy file.
    """
    save_dir = "/home/users/slahlali/diod_3d/data/waymo_front/masks_fus/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, sample_path.replace(".jpg", ".npz"))
    np.savez_compressed(save_path, data=pred.cpu().numpy())


def voc_ap(rec, prec, use_07_metric=False):
    """
    Computes VOC AP given precision and recall.
    """
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_iou(mask1, set_masks):
    """
    Calculates IoU between a single mask and a set of masks.
    """
    mask1 = np.array(mask1, dtype=bool)
    iou_scores = []
    for mask2 in set_masks:
        mask2 = np.array(mask2, dtype=bool)
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / (np.sum(union) + 1e-10)
        iou_scores.append(iou)
    return iou_scores
