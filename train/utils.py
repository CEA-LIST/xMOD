import os
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from torch.nn.parameter import Parameter
from tqdm import tqdm
from eval.metrics import ARI # Assuming ARI is in this path

# --- Core Utilities ---

def set_seeds(seed):
    """Sets random seeds for all relevant libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed {seed} set for reproducibility.")


def load_model(model, state_dict):
    """Loads a state dictionary into a model, intelligently handling key mismatches."""
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(f"Warning: Key '{name}' from checkpoint not found in model. Skipping.")
            continue
        if isinstance(param, Parameter):
            param = param.data
        if own_state[name].data.shape != param.shape:
            print(f"Warning: Shape mismatch for '{name}'. Skipping.")
            continue
        own_state[name].copy_(param)
    print("Model loaded successfully.")


def save_checkpoint(epoch, model, filepath):
    """Saves a model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, filepath)
    print(f"Checkpoint saved to {filepath}")


# --- Teacher-Student Utilities ---

@torch.no_grad()
def update_teacher_model(student_model, teacher_model, keep_rate=0.996):
    """Performs an Exponential Moving Average (EMA) update of the teacher model's weights."""
    student_dict = student_model.state_dict()
    teacher_dict = teacher_model.state_dict()
    new_teacher_dict = OrderedDict()
    for key, value in teacher_dict.items():
        if key in student_dict:
            new_teacher_dict[key] = (student_dict[key] * (1 - keep_rate)) + (value * keep_rate)
        else:
            raise KeyError(f"Key '{key}' not found in student model during teacher update.")
    teacher_model.load_state_dict(new_teacher_dict)


@torch.no_grad()
def process_teacher_predictions(opt, masks_t, mask_fg_t):
    """
    Processes raw teacher model output to generate high-quality pseudo-labels.
    This unified function handles quantization, component splitting, and filtering.
    """
    masks_t = masks_t.clone() * 0.999 + 1e-8
    masks_t_quant = masks_t.argmax(dim=2).cpu().numpy()
    mask_fg_t_np = mask_fg_t.cpu().numpy()
    
    # The last slot is reserved for the background
    masks_t_quant[masks_t_quant == (opt.num_slots - 1)] = 0

    for i in range(opt.batch_size):
        for j in range(masks_t_quant.shape[1]):  # Iterate through frames
            activated_slots = list(np.unique(masks_t_quant[i, j]))
            if 0 in activated_slots: activated_slots.remove(0)
            
            empty_slots = [s for s in range(1, opt.num_slots) if s not in activated_slots]
            
            for slot_idx in activated_slots:
                binary_mask = (masks_t_quant[i, j] == slot_idx)
                labeled_mask, num_features = label(binary_mask)
                
                if num_features > 1:  # Slot has multiple disconnected components
                    for feat_idx in range(2, num_features + 1):
                        component_mask = (labeled_mask == feat_idx)
                        is_confident = mask_fg_t_np[i, j, 0][component_mask].mean() > opt.teacher_confidence_threshold
                        is_large_enough = component_mask.sum() > 10

                        if is_confident and is_large_enough and empty_slots:
                            masks_t_quant[i, j][component_mask] = empty_slots.pop(0)
                        else:
                            masks_t_quant[i, j][component_mask] = 0  # Assign to background

    # Filter out very small, noisy components
    h, w = masks_t_quant.shape[-2:]
    for i in range(opt.batch_size):
        for j in range(masks_t_quant.shape[1]):
            values, indices, counts = np.unique(masks_t_quant[i, j], return_inverse=True, return_counts=True)
            mapping = np.arange(len(values))
            mapping[counts <= 10] = 0  # Map small components to background
            masks_t_quant[i, j] = mapping[indices].reshape((h, w))

    # Convert to one-hot tensor for the loss function
    masks_t_bin = torch.from_numpy(masks_t_quant).long()
    n_objects_t = masks_t_bin.max()
    
    if n_objects_t > 0:
        masks_t_one_hot = F.one_hot(masks_t_bin, n_objects_t + 1)[..., 1:].permute(0, 1, 4, 2, 3).float()
    else:
        masks_t_one_hot = torch.empty(opt.batch_size, masks_t.shape[1], 0, h, w)

    return masks_t_one_hot, n_objects_t


# --- Visualization & Evaluation ---

@torch.no_grad()
def visualize_kitti(epoch, model, test_set, opt, sample_path, mode, device):
    """Generates and saves visualizations for the KITTI dataset."""
    model.eval()
    sample = test_set[0]
    data_key = 'range' if mode == '3d' else 'image'
    data = sample[data_key].unsqueeze(0).to(device)
    
    recon_combined, masks, _, _ = model(data)

    data_resized = F.interpolate(data[0].unsqueeze(0), opt.mask_resolution)[0]
    recon_combined = F.interpolate(recon_combined[0].unsqueeze(0), opt.mask_resolution)[0]
    masks = masks[0]

    n_cols = opt.num_slots + 2
    fig, ax = plt.subplots(1, n_cols, figsize=(n_cols * 3, 3))
    
    image_vis = data_resized.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    recon_vis = recon_combined.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    
    vis_channel = image_vis[:, :, 3] if mode == '3d' else image_vis
    recon_channel = recon_vis[:, :, 3] if mode == '3d' else recon_vis

    ax[0].imshow(np.clip(vis_channel, 0, 1))
    ax[0].set_title('Original')
    ax[1].imshow(np.clip(recon_channel, 0, 1))
    ax[1].set_title('Reconstruction')

    for j in range(opt.num_slots):
        ax[j + 2].imshow(np.clip(vis_channel, 0, 1))
        ax[j + 2].imshow(masks[0, j].cpu().numpy(), cmap='viridis', alpha=0.6)
        ax[j + 2].set_title(f'Slot {j}')

    for axis in ax: axis.axis('off')
    eval_name = os.path.join(sample_path, f'epoch_{epoch}_slots_{mode}.png')
    fig.tight_layout()
    fig.savefig(eval_name)
    plt.close(fig)
    print(f"Saved KITTI visualization to {eval_name}")


@torch.no_grad()
def visualize_pd(epoch, model, test_set, opt, sample_path, mode, device):
    """Generates and saves visualizations for the PD dataset."""
    model.eval()
    sample = test_set[0]
    data_key = 'range' if mode == '3d' else 'image'
    data = sample[data_key][:5].unsqueeze(0).to(device) # PD uses 5 frames for vis

    recon, masks, mask_fg, _ = model(data)
    
    image_vis = F.interpolate(data[0], opt.mask_resolution)
    recon_vis = F.interpolate(recon[0], opt.mask_resolution)

    fig, ax = plt.subplots(1, opt.num_slots + 2, figsize=((opt.num_slots + 2) * 3, 3))
    
    img_np = (image_vis[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    recon_np = (recon_vis[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)

    ax[0].imshow(img_np)
    ax[0].set_title('Original')
    ax[1].imshow(recon_np)
    ax[1].set_title('Reconstruction')

    for j in range(opt.num_slots):
        ax[j + 2].imshow(img_np)
        ax[j + 2].imshow(masks[0, 0, j].cpu().numpy(), cmap='viridis', alpha=0.6)
        ax[j + 2].set_title(f'Slot {j + 1}')

    for axis in ax: axis.axis('off')
    eval_name = os.path.join(sample_path, f'epoch_{epoch}_slots_{mode}.png')
    fig.tight_layout()
    fig.savefig(eval_name)
    plt.close(fig)
    print(f"Saved PD visualization to {eval_name}")


@torch.no_grad()
def evaluate_ari(model, test_set, device, num_slots):
    """Evaluates the model on the test set and returns the mean ARI score (for PD dataset)."""
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    fg_aris = []

    for sample in tqdm(test_loader, desc="Evaluating ARI"):
        image = sample['image'].to(device)
        mask_gt = F.interpolate(sample['mask'].float(), (120, 242)).long()

        for i in range(image.size(1) // 5):
            img_chunk = image[:, i*5:(i+1)*5]
            gt_chunk = mask_gt[:, i*5:(i+1)*5]

            _, masks, _, _ = model(img_chunk)
            masks = masks.cpu()

            gt_flat = gt_chunk[0].flatten()
            pred_flat = masks[0].argmax(dim=1).flatten()

            gt_one_hot = F.one_hot(gt_flat).unsqueeze(0)
            pred_one_hot = F.one_hot(pred_flat, num_classes=num_slots).unsqueeze(0)

            if gt_one_hot.shape[1] <= 2: continue

            fg_ari_score = ARI(gt_one_hot[:, 1:, :], pred_one_hot)
            if not torch.isnan(fg_ari_score):
                fg_aris.append(fg_ari_score.item())
    
    mean_fg_ari = np.mean(fg_aris) if fg_aris else 0.0
    print(f"Evaluation Results - FG ARI: {mean_fg_ari:.4f}")
    return mean_fg_ari
