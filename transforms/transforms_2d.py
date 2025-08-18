# transforms/transforms_2d.py

from __future__ import annotations
from typing import Callable
import torch
from torch import Tensor, nn
from torchaug.batch_transforms import (
    BatchVideoWrapper,
    BatchRandomColorJitter,
    BatchRandomGaussianBlur,
    BatchRandomGrayScale,
    BatchRandomSolarize
)
from torchaug.transforms import VideoNormalize
from torchvision.transforms import Compose, InterpolationMode, RandomResizedCrop, RandomHorizontalFlip
import torchvision.transforms.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomHorizontalFlipSample(RandomHorizontalFlip):
    """Applies horizontal flip to both image and mask in a sample dictionary."""
    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if torch.rand(1) < self.p:
            sample["image"] = F.hflip(sample["image"])
            sample["mask"] = F.hflip(sample["mask"])
        return sample

class RandomResizedCropSample(RandomResizedCrop):
    """Applies random resized crop to both image and mask in a sample dictionary."""
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), 
                 interpolation=InterpolationMode.BILINEAR, antialias=True, 
                 interpolation_mask=InterpolationMode.NEAREST, p: float = 0.5):
        super().__init__(size, scale, ratio, interpolation, antialias=antialias)
        self.p = p
        self.interpolation_mask = interpolation_mask

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if torch.rand(1) < self.p:
            i, j, h, w = self.get_params(sample["image"], self.scale, self.ratio)
            sample["image"] = F.resized_crop(
                sample["image"], i, j, h, w, self.size, self.interpolation, antialias=self.antialias
            )
            sample["mask"] = F.resized_crop(
                sample["mask"], i, j, h, w, self.size, self.interpolation_mask, antialias=self.antialias
            )
        return sample

def get_2d_transforms(opt):
    """
    Returns a set of 2D data transformations for the train, student, and teacher pipelines.
    """
    # Geometric transforms applied on CPU before batching
    train_transform = Compose([
        RandomResizedCropSample(
            size=(opt.resolution[0], opt.resolution[1]),
            scale=opt.crop_scale,
            ratio=opt.crop_ratio,
            interpolation=InterpolationMode.BILINEAR,
            interpolation_mask=InterpolationMode.NEAREST,
            p=opt.p_crop
        ),
        RandomHorizontalFlipSample(p=opt.p_flip)
    ])

    # Photometric transforms for the student model (applied on GPU)
    student_transform = BatchVideoWrapper([
        BatchRandomColorJitter(
            opt.brightness, opt.contrast, opt.saturation, opt.hue, opt.p_contrastive
        ),
        BatchRandomGrayScale(opt.p_grayscale),
        VideoNormalize(opt.mean, opt.std, video_format="TCHW")
    ], inplace=False, same_on_frames=True, video_format="TCHW").to(device)

    # Simple normalization for the teacher model (applied on GPU)
    teacher_transform = VideoNormalize(
        opt.mean, opt.std, inplace=False, video_format="TCHW"
    ).to(device)

    return train_transform, student_transform, teacher_transform

def get_crop_transform(
    size: int | list[int] | None = None,
    scale: list[float] = (0.08, 1.0),
    ratio: list[float] = (3.0 / 4.0, 4.0 / 3.0),
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    interpolation_mask: InterpolationMode = InterpolationMode.NEAREST,
    p_crop: float = 0.,
    p_flip: float = 0.5,
) -> Callable:
    return Compose(
        [
            RandomResizedCropSample(size, scale, ratio, interpolation, True, interpolation_mask, p_crop),
            RandomHorizontalFlipSample(p_flip)
        ]
    )


def get_ssl_online_transform(
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    hue: float = 0.1,
    p_contrastive: float = 0.8,
    p_grayscale: float = 0.2,
    kernel_size_blur: int = 23,
    sigma_blur: list[int] = [0.1, 2.],
    p_blur: float = 0.5,
    threshold_solarize: float = 0.5,
    p_solarize: float = 0.2,
    mean: list[int] = [0.485, 0.456, 0.406],
    std: list[int] = [0.229, 0.224, 0.225]
) -> nn.Module:
    return BatchVideoWrapper(
        [
            BatchRandomColorJitter(brightness, contrast, saturation, hue, p_contrastive),
            BatchRandomGrayScale(p_grayscale),
            BatchRandomGaussianBlur(kernel_size_blur, sigma_blur, p_blur),
            BatchRandomSolarize(threshold_solarize, p_solarize),
            VideoNormalize(mean, std, video_format="TCHW")
        ],
        inplace=False,
        same_on_frames=True,
        video_format="TCHW",
    )


def get_normalize_transform(
    mean: list[int] = [0.485, 0.456, 0.406],
    std: list[int] = [0.229, 0.224, 0.225]
) -> nn.Module:
    return VideoNormalize(mean, std, inplace=False, video_format="TCHW")