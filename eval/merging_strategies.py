import torch

def merge_predictions(pred2D, pred3D, iou_threshold=0.1):
    """
    Merges 2D and 3D predictions based on IoU.
    """
    indices1 = torch.argmax(pred2D, dim=1)
    indices2 = torch.argmax(pred3D, dim=1)

    mask1 = torch.zeros_like(pred2D, dtype=torch.bool)
    mask2 = torch.zeros_like(pred3D, dtype=torch.bool)

    for i in range(pred2D.shape[1]):
        mask1[:, i, :, :] = indices1 == i
        mask2[:, i, :, :] = indices2 == i

    final_mask = torch.zeros_like(mask1, dtype=torch.bool)
    occupied = torch.zeros((1, pred2D.shape[1]), dtype=torch.bool)

    for i in range(pred2D.shape[1]):
        for j in range(pred3D.shape[1]):
            intersection = (mask1[:, i, :, :] & mask2[:, j, :, :]).float().sum()
            union = (mask1[:, i, :, :] | mask2[:, j, :, :]).float().sum()
            iou = intersection / (union + 1e-10)
            if iou > iou_threshold:
                final_mask[:, i, :, :] |= mask1[:, i, :, :] | mask2[:, j, :, :]
                occupied[:, i] = True

    return final_mask.to(torch.int)
