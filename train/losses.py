import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_CLAMP_MIN = 1e-8
MASK_MULTIPLIER = 0.999

def compute_matching_scores(opt, pred_masks, gt_masks_one_hot, n_gt_objects):
    """Computes matching scores (cross-entropy) between predicted and GT masks."""
    mask_detach = (pred_masks.detach().flatten(3, 4) * MASK_MULTIPLIER + LOG_CLAMP_MIN).cpu().numpy()
    gt_masks_np = gt_masks_one_hot.flatten(3, 4).cpu().numpy()
    scores = np.zeros((opt.batch_size, pred_masks.shape[1], opt.num_slots, n_gt_objects))

    for i in range(opt.batch_size):
        for j in range(pred_masks.shape[1]):
            log_pred = np.log(mask_detach[i, j])
            log_one_minus_pred = np.log(1 - mask_detach[i, j])
            cross_entropy = np.matmul(log_pred, gt_masks_np[i, j].T) + \
                            np.matmul(log_one_minus_pred, (1 - gt_masks_np[i, j]).T)
            scores[i, j] = cross_entropy
    return scores

def compute_weighted_bce(opt, pred_masks, gt_masks, scores, device, teacher_fg_mask=None):
    """
    Computes Weighted BCE loss after finding the optimal matching via the Hungarian algorithm.
    """
    scores = scores[:, :, :-1, :] # Discard background slot
    pred_masks = pred_masks * MASK_MULTIPLIER + LOG_CLAMP_MIN
    total_mask_loss = 0.0
    filtered_gt_fg = torch.zeros_like(gt_masks[:, :, 0, :, :], device=device) if teacher_fg_mask is not None else None

    for i in range(opt.batch_size):
        for j in range(pred_masks.shape[1]):
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-scores[i, j])
            batch_frame_loss = 0.0
            num_valid_objects = 0

            for pred_idx, gt_idx in zip(row_ind, col_ind):
                current_gt_mask = gt_masks[i, j, gt_idx, :, :]
                if not current_gt_mask.max(): continue

                if teacher_fg_mask is not None:
                    confidence = teacher_fg_mask[i, j, 0, :, :][current_gt_mask == 1].mean()
                    if confidence >= opt.teacher_confidence_threshold:
                        weight = 1.0 + confidence
                        num_valid_objects += 1
                        if filtered_gt_fg is not None:
                            filtered_gt_fg[i, j, :, :] += current_gt_mask
                    else: continue
                else: # Burn-in mode
                    fg_portion = current_gt_mask.sum() / current_gt_mask.numel()
                    weight = 2.0 - fg_portion
                    num_valid_objects += 1

                bce_term = -weight * torch.log(pred_masks[i, j, pred_idx]) * current_gt_mask
                bce_term -= (1 - current_gt_mask) * torch.log(1 - pred_masks[i, j, pred_idx])
                batch_frame_loss += bce_term.mean()

            if num_valid_objects > 0:
                total_mask_loss += batch_frame_loss / num_valid_objects

    return total_mask_loss / opt.batch_size, filtered_gt_fg

def compute_loss(opt, batch, outputs, epoch, device):
    """Computes the total loss for the current training step."""
    losses = {}
    total_loss = torch.tensor(0.0, device=device)

    # --- 2D Losses ---
    if '2d' in opt.mode:
        recon_loss_2d = nn.MSELoss()(outputs['recon_2d'], batch['student_image_2d'])
        losses['recon_loss_2d'] = recon_loss_2d
        total_loss += opt.weight_recon * recon_loss_2d

        gt_mask_2d = batch['mask_gt_2d']
        n_objects_2d = gt_mask_2d.max()
        if n_objects_2d > 0:
            gt_mask_2d_one_hot = F.one_hot(gt_mask_2d, n_objects_2d + 1)[:, :, :, :, 1:].permute(0, 1, 4, 2, 3).float()
            scores_2d = compute_matching_scores(opt, outputs['masks_2d'], gt_mask_2d_one_hot, n_objects_2d)
            teacher_fg = outputs.get('teacher_mask_fg_3d')
            mask_loss_2d, filtered_gt = compute_weighted_bce(opt, outputs['masks_2d'], gt_mask_2d_one_hot, scores_2d, device, teacher_fg)
            losses['mask_loss_2d'] = mask_loss_2d
            total_loss += opt.weight_mask * mask_loss_2d
            
            mask_fg_2d = outputs['mask_fg_2d'] * MASK_MULTIPLIER + LOG_CLAMP_MIN
            gt_fg = filtered_gt if filtered_gt is not None else gt_mask_2d_one_hot.sum(dim=2)
            nll_loss_2d = (-torch.log(mask_fg_2d[:, :, 0]) * gt_fg).mean()
            reg_loss_2d = mask_fg_2d[:, :, 0].mean()
            losses.update({'nll_loss_2d': nll_loss_2d, 'reg_loss_2d': reg_loss_2d})
            total_loss += opt.weight_nll * nll_loss_2d + opt.weight_reg * reg_loss_2d

        if 'ts' in opt.mode and epoch >= opt.start_teacher_epoch and 'teacher_masks_2d' in outputs:
            teacher_masks = outputs['teacher_masks_2d']
            if teacher_masks.shape[2] > 0:
                scores_ts = compute_matching_scores(opt, outputs['masks_2d'], teacher_masks, outputs['teacher_n_objects_2d'])
                ts_loss_2d, _ = compute_weighted_bce(opt, outputs['masks_2d'], teacher_masks.to(device), scores_ts, device, outputs['teacher_mask_fg_2d'])
                losses['ts_loss_2d'] = ts_loss_2d
                total_loss += ts_loss_2d

    # --- 3D Losses ---
    if '3d' in opt.mode:
        valid_points = batch['student_range_3d'] > 0
        recon_loss_3d = (nn.MSELoss(reduction='none')(outputs['recon_3d'], batch['student_range_3d']) * valid_points).sum() / valid_points.sum()
        losses['recon_loss_3d'] = recon_loss_3d
        total_loss += opt.weight_recon * recon_loss_3d

        gt_mask_3d = batch['mask_gt_3d']
        n_objects_3d = gt_mask_3d.max()
        if n_objects_3d > 0:
            gt_mask_3d_one_hot = F.one_hot(gt_mask_3d, n_objects_3d + 1)[:, :, :, :, 1:].permute(0, 1, 4, 2, 3).float()
            scores_3d = compute_matching_scores(opt, outputs['masks_3d'], gt_mask_3d_one_hot, n_objects_3d)
            teacher_fg = outputs.get('teacher_mask_fg_2d')
            mask_loss_3d, filtered_gt_3d = compute_weighted_bce(opt, outputs['masks_3d'], gt_mask_3d_one_hot, scores_3d, device, teacher_fg)
            losses['mask_loss_3d'] = mask_loss_3d
            total_loss += opt.weight_mask * mask_loss_3d

            mask_fg_3d = outputs['mask_fg_3d'] * MASK_MULTIPLIER + LOG_CLAMP_MIN
            gt_fg_3d = filtered_gt_3d if filtered_gt_3d is not None else gt_mask_3d_one_hot.sum(dim=2)
            nll_loss_3d = (-torch.log(mask_fg_3d[:, :, 0]) * gt_fg_3d).mean()
            reg_loss_3d = mask_fg_3d[:, :, 0].mean()
            losses.update({'nll_loss_3d': nll_loss_3d, 'reg_loss_3d': reg_loss_3d})
            total_loss += opt.weight_nll * nll_loss_3d + opt.weight_reg * reg_loss_3d

        if 'ts' in opt.mode and epoch >= opt.start_teacher_epoch and 'teacher_masks_3d' in outputs:
            teacher_masks = outputs['teacher_masks_3d']
            if teacher_masks.shape[2] > 0:
                scores_ts_3d = compute_matching_scores(opt, outputs['masks_3d'], teacher_masks, outputs['teacher_n_objects_3d'])
                ts_loss_3d, _ = compute_weighted_bce(opt, outputs['masks_3d'], teacher_masks.to(device), scores_ts_3d, device, outputs['teacher_mask_fg_3d'])
                losses['ts_loss_3d'] = ts_loss_3d
                total_loss += ts_loss_3d
    
    losses['total_loss'] = total_loss
    return losses
