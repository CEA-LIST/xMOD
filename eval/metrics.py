import numpy as np
import torch
import torch.nn.functional as F
from eval.metrics import ARI
from utils_eval import voc_ap, calculate_iou


def ARI(true_mask, pred_mask):
    "adjusted_rand_index"
    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    assert not (n_points <= n_true_groups and n_points <= n_pred_groups), ("adjusted_rand_index requires n_groups < n_points. We don't handle the special cases that can occur when you have one cluster per datapoint.")

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    true_mask_oh = true_mask.to(torch.float32) 
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).to(torch.float32)

    n_points = torch.sum(true_mask_oh, dim=[1, 2]).to(torch.float32)

    nij = torch.einsum('bji,bjk->bki', pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)

    rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), dim=1)
    bindex = torch.sum(b * (b - 1), dim=1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex+0.000000000001)

    _all_equal = lambda values: torch.all(torch.eq(values, values[..., :1]), dim=-1)
    both_single_cluster = torch.logical_and(_all_equal(true_group_ids), _all_equal(pred_group_ids))
    
    return torch.where(both_single_cluster, torch.ones_like(ari), ari)

def calculate_ari_scores(gt_mask, pred_mask, num_slots):
    """
    Calculates fg-ARI and all-ARI scores.
    """
    gt_mask_flat = gt_mask.view(-1)
    pred_mask_flat = pred_mask.view(num_slots, -1).permute(1, 0)
    gt_one_hot = F.one_hot(gt_mask_flat)

    if gt_one_hot.shape[1] <= 2:
        return None, None

    all_ari = ARI(gt_one_hot.unsqueeze(0), pred_mask_flat.unsqueeze(0))
    fg_ari = ARI(gt_one_hot[:, 1:].unsqueeze(0), pred_mask_flat.unsqueeze(0))

    return fg_ari, all_ari


def voc_eval(gt_masks, pred_scores, pred_masks, ovthresh=0.5, use_07_metric=False):
    """
    Performs PASCAL VOC evaluation.
    """
    nb_images = len(pred_masks)
    image_ids = []
    class_recs = {}
    nb_gt = 0

    for im in range(nb_images):
        image_ids += [im] * len(pred_masks[im])
        class_recs[im] = [False] * len(gt_masks[im])
        nb_gt += len(gt_masks[im])

    pred_scores_flat = np.array(
        [item for sublist in pred_scores for item in sublist]
    )
    pred_masks_flat = np.stack([item for sublist in pred_masks for item in sublist])

    sorted_ind = np.argsort(-pred_scores_flat)
    pred_masks_flat = pred_masks_flat[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        mask = pred_masks_flat[d]
        gt_masks_for_image = gt_masks[image_ids[d]]
        overlaps = calculate_iou(mask, gt_masks_for_image)

        ovmax = np.max(overlaps) if overlaps else 0
        jmax = np.argmax(overlaps) if overlaps else -1

        if ovmax >= ovthresh:
            if not R[jmax]:
                tp[d] = 1.0
                R[jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(nb_gt)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def calculate_f1_score(gt_masks, scores, pred_masks, conf_threshold=0.5):
    """
    Calculates the F1 score.
    """
    rec, prec, _ = voc_eval(gt_masks, scores, pred_masks, conf_threshold)
    precision = prec[-1] if len(prec) > 0 else 0
    recall = rec[-1] if len(rec) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1, precision, recall
