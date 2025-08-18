# transforms/transforms_3d.py

from __future__ import annotations
import torch
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode, RandomResizedCrop
import torchvision.transforms.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModuleCompose(nn.Module):
    """A nn.Module version of Compose for sequential transformations."""
    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class BatchRandomHorizontalFlip3D(nn.Module):
    """Applies horizontal flip to 3D range data and masks."""
    def __init__(self, p: float, bs: int):
        super().__init__()
        self.p = p
        self.bs = bs
    
    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if torch.rand(1) < self.p:
            for b in range(self.bs):
                sample["range"][b] = F.hflip(sample["range"][b])
                # Invert the horizontal coordinate (y-channel in spherical projection)
                values = sample["range"][b][:, 1]
                mask = values > 0
                sample["range"][b][:, 1][mask] = 1 - values[mask]
                sample["mask"][b] = F.hflip(sample["mask"][b])
        return sample

class BatchDataJitter3D(nn.Module):
    """Applies random jitter to the 3D point coordinates."""
    def __init__(self, p: float, bs: int):
        super().__init__()
        self.p = p
        self.bs = bs
    
    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if torch.rand(1) < self.p:
            for b in range(self.bs):
                jitter = torch.tensor([
                    torch.empty(1).uniform_(0, 0.1),
                    torch.empty(1).uniform_(0, 0.1),
                    torch.empty(1).uniform_(-0.1, 0.1)
                ]).to(sample["range"].device)
                
                mask = sample["range"][b][:, 3] > 0
                sample["range"][b, :, :3, :, :][mask] += jitter.view(1, 3, 1, 1)
                sample["range"][b, :, :3, :, :] = torch.clamp(sample["range"][b, :, :3, :, :], 0, 1)
                
                # Recalculate depth/norm
                new_norm = torch.norm(sample["range"][b, :, :3, :, :], p=2, dim=1)
                sample["range"][b, :, 3, :, :][mask] = new_norm[mask]
        return sample

class BatchDropPoints3D(nn.Module):
    """Randomly drops a fraction of points from the projection."""
    def __init__(self, p: float, bs: int):
        super().__init__()
        self.p = p
        self.bs = bs
    
    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if torch.rand(1) < self.p:
            total_points = sample["range"].shape[-1] * sample["range"].shape[-2]
            num_to_drop = int(total_points * self.p)
            
            for b in range(self.bs):
                indices_to_drop = torch.randperm(total_points)[:num_to_drop]
                mask = torch.ones(total_points, dtype=torch.bool, device=sample["range"].device)
                mask[indices_to_drop] = False
                mask = mask.view(sample["range"].shape[-2], sample["range"].shape[-1])
                sample["range"][b, :, ~mask] = 0
        return sample

def get_3d_transforms(opt):
    """
    Returns a set of 3D data transformations for training.
    In the 3D case, student and train transforms are the same.
    """
    train_transform = ModuleCompose([
        # Note: 3D ResizedCrop is complex and omitted for clarity unless essential
        BatchRandomHorizontalFlip3D(p=opt.p_flip, bs=opt.batch_size),
        BatchDropPoints3D(p=opt.p_drop, bs=opt.batch_size),
        BatchDataJitter3D(p=opt.p_jitter, bs=opt.batch_size)
    ])
    
    student_transform = train_transform.to(device)
    
    return train_transform, student_transform
