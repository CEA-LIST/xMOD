from __future__ import annotations

import os
import random
import json
import math
from typing import Callable, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np


BANNED_SCENES = [
    'scene_000100', 'scene_000002', 'scene_000008', 'scene_000012', 'scene_000018', 'scene_000029',
    'scene_000038', 'scene_000040', 'scene_000043', 'scene_000044', 'scene_000049', 'scene_000050',
    'scene_000053', 'scene_000063', 'scene_000079', 'scene_000090', 'scene_000094', 'scene_000100',
    'scene_000103', 'scene_000106', 'scene_000111', 'scene_000112', 'scene_000124', 'scene_000125',
    'scene_000127', 'scene_000148', 'scene_000159', 'scene_000166', 'scene_000169', 'scene_000170',
    'scene_000171', 'scene_000187', 'scene_000191', 'scene_000200', 'scene_000202', 'scene_000217',
    'scene_000218', 'scene_000225', 'scene_000229', 'scene_000232', 'scene_000236', 'scene_000237',
    'scene_000245', 'scene_000249'
]

CAMERA_VIEWS = [1, 5, 6, 7, 8, 9]



class BasePDDataset(Dataset):
    """
    An abstract base class for the Pandaset dataset variants.

    This class handles common functionalities such as file listing, transformations,
    and data processing helpers, providing a solid foundation for specific
    dataset implementations. Subclasses must implement `_setup_paths` and
    `__getitem__`.
    """
    def __init__(
        self,
        root: str,
        split: str,
        downsampling_ratio: float = 0.5,
        crop: int = 128
    ):
        """
        Initializes the base dataset.

        Args:
            root (str): The root directory of the Pandaset data.
            split (str): The data split to use (e.g., 'train', 'eval', 'test').
            downsampling_ratio (float): The ratio to downsample images.
            crop (int): The number of pixels to crop from the top of the images.
        """
        super().__init__()
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Root directory not found: {root}")

        self.root_dir = root
        self.split = split
        self.downsampling_ratio = downsampling_ratio
        self.crop = crop

        self.files = self._get_split_files()
        self.img_transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # These lists will be populated by the `_setup_paths` method in subclasses.
        self.real_files: List[str] = []
        self.mask_files: List[str] = []
        self.depth_files: List[str] = []
        self.calibration_file: str = ''

        self._setup_paths()

    def _get_split_files(self) -> List[str]:
        """
        Gets the scene files from the root directory corresponding to the split.
        This method can be overridden by subclasses for custom split logic.
        """
        all_files = sorted(os.listdir(self.root_dir))
        if self.split == 'train':
            return all_files[1:]
        elif self.split == 'eval':
            return all_files[0:1]
        elif self.split == 'test':
             return all_files
        else: # 'all' or a custom split name
            return all_files

    def _setup_paths(self):
        """
        Abstract method to populate the file path lists (real_files, mask_files, etc.).
        This method must be implemented by subclasses to define data sources.
        """
        raise NotImplementedError("Subclasses must implement the `_setup_paths` method.")

    def _process_image_and_mask(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies common resizing and cropping to an image-mask pair.
        """
        width = int(math.ceil(image.shape[1] * self.downsampling_ratio))
        height = int(math.ceil(image.shape[0] * self.downsampling_ratio))
        dim = (width, height)

        image_processed = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)[self.crop:, :, :]
        mask_processed = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)[self.crop:, :]

        return image_processed, mask_processed

    def _remap_mask(self, mask: np.ndarray, threshold: int) -> np.ndarray:
        """
        Remaps mask instance IDs to be contiguous (1, 2, 3...), filtering out
        small segments below a specified pixel count threshold.
        """
        mapping = {0: 0}
        mapping_count = 1
        values, indices, counts = np.unique(mask, return_inverse=True, return_counts=True)

        for i, value in enumerate(values):
            if value != 0 and counts[i] > threshold:
                mapping[value] = mapping_count
                mapping_count += 1

        # Create a mapping array and use it to efficiently remap the mask
        cur_mapping = np.array([mapping.get(v, 0) for v in values])
        remapped_mask = cur_mapping[indices].reshape(mask.shape)
        return remapped_mask

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.real_files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Abstract method to retrieve a data sample.
        This must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the `__getitem__` method.")




class PDDataset(BasePDDataset):
    """
    Dataset for loading 2D image and mask sequences from Pandaset.
    This class corresponds to the original `datasetPD.py`.
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        supervision: str = 'moving',
        transform: Callable = None,
        apply_img_transform: bool = True
    ):
        self.supervision = supervision
        self.transform = transform
        self.apply_image_transform = apply_img_transform
        super().__init__(root, split)

    def _setup_paths(self):
        """Sets up paths for RGB images and masks based on supervision type."""
        annotation_map = {
            'moving': 'moving_masks',
            'all': 'ari_masks',
            'est': 'est_masks'
        }
        if self.supervision not in annotation_map:
            raise ValueError(f"Invalid supervision type: {self.supervision}. Choose from {list(annotation_map.keys())}")
        annotation_dir = annotation_map[self.supervision]

        for f in self.files:
            if f in BANNED_SCENES:
                continue
            for i in CAMERA_VIEWS:
                camera_path = f'/rgb/camera_0{i}'
                if os.path.exists(os.path.join(self.root_dir, f + camera_path)):
                    self.real_files.append(f + camera_path)
                    self.mask_files.append(f'/{annotation_dir}/camera_0{i}')

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Loads a random sequence of 5 consecutive frames (image and mask)."""
        path = self.real_files[index]
        mask_path = self.mask_files[index]
        all_images = sorted(os.listdir(os.path.join(self.root_dir, path)))

        rand_id = random.randint(0, len(all_images) - 6)
        frame_indices = [rand_id + j for j in range(5)]

        ims, masks = [], []
        for idd in frame_indices:
            image = cv2.imread(os.path.join(self.root_dir, path, all_images[idd]))
            mask = cv2.imread(os.path.join(self.root_dir, mask_path, all_images[idd]), cv2.IMREAD_UNCHANGED)

            image, mask = self._process_image_and_mask(image, mask)
            mask = self._remap_mask(mask, threshold=50)

            ims.append(torch.from_numpy(image).float())
            masks.append(torch.from_numpy(mask).long())

        ims_tensor = torch.stack(ims).permute(0, 3, 1, 2) / 255.0
        masks_tensor = torch.stack(masks)

        sample = {'image': ims_tensor, 'mask': masks_tensor}

        if self.transform:
            sample = self.transform(sample)
        elif self.apply_image_transform:
            sample["image"] = self.img_transform(sample["image"])

        return sample



class PDDataset3DOD(BasePDDataset):
    """
    Dataset for loading 3D data (image, mask, depth/range) from Pandaset.
    This class corresponds to the original `datasetPD_3DOD.py`.
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        supervision: str = 'moving',
        transform: Callable = None,
        apply_img_transform: bool = False
    ):
        self.supervision = supervision
        self.transform = transform
        self.apply_image_transform = apply_img_transform
        super().__init__(root, split)
    
    def _get_split_files(self) -> List[str]:
        """Overrides base method for the specific split logic of this dataset."""
        all_files = sorted(os.listdir(self.root_dir))
        if self.split == 'train':
            return all_files[18:] # Use scenes starting from the 18th for training
        return super()._get_split_files()

    def _setup_paths(self):
        """Sets up paths for RGB, masks, and depth data."""
        annotation_map = {
            'moving': 'moving_masks',
            'all': 'ari_masks',
            'est': 'est_masks'
        }
        if self.supervision not in annotation_map:
            raise ValueError(f"Invalid supervision type: {self.supervision}. Choose from {list(annotation_map.keys())}")
        annotation_dir = annotation_map[self.supervision]

        for f in self.files:
            if f in BANNED_SCENES:
                continue
            for i in CAMERA_VIEWS:
                camera_path = f'/rgb/camera_0{i}'
                if os.path.exists(os.path.join(self.root_dir, f + camera_path)):
                    self.real_files.append(f + camera_path)
                    self.mask_files.append(f'/{annotation_dir}/camera_0{i}')
                    self.depth_files.append(f'/depth/camera_0{i}')
                    self.calibration_file = os.path.join(self.root_dir, f, 'calibration/f7ac1d5879108bf162666fa9836d1f9ea712a35f.json')

    def _depth2point(self, depth_map: np.ndarray, camera: str) -> np.ndarray:
        """Converts a depth map to a 4-channel point cloud representation (x, y, z, range)."""
        with open(self.calibration_file) as f:
            calib = json.load(f)

        cam_idx = calib['names'].index(camera)
        intrinsic = calib['intrinsics'][cam_idx]
        
        fx, cx, fy, cy = intrinsic['fx'], intrinsic['cx'], intrinsic['fy'], intrinsic['cy']
        K_inv = np.linalg.inv(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))

        h, w = depth_map.shape
        z = np.where(depth_map > 70, 0, depth_map) # Clip depth at 70m

        v, u = np.mgrid[0:h, 0:w]
        uv1 = np.stack([u.flatten(), v.flatten(), np.ones(h*w)], axis=0)

        in_camera_coord = (K_inv @ uv1) * z.flatten()
        in_camera_coord = in_camera_coord.T.reshape(h, w, 3)
        
        x, y, z = in_camera_coord[..., 0], in_camera_coord[..., 1], in_camera_coord[..., 2]

        # Normalize channels for model consumption
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        z_norm = (z - z.min()) / (70 - z.min() + 1e-8) # Normalize with global max
        
        range_channel = np.linalg.norm(in_camera_coord, axis=2)
        range_norm = (range_channel - range_channel.min()) / (range_channel.max() - range_channel.min() + 1e-8)

        return np.stack([x_norm, y_norm, z_norm, range_norm], axis=-1)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Loads a random sequence of 5 consecutive frames (image, mask, range)."""
        path = self.real_files[index]
        mask_path = self.mask_files[index]
        depth_path = self.depth_files[index]
        all_images = sorted(os.listdir(os.path.join(self.root_dir, path)))

        rand_id = random.randint(0, len(all_images) - 6)
        frame_indices = [rand_id + j for j in range(5)]

        ims, masks, ranges = [], [], []
        for idd in frame_indices:
            img_file = all_images[idd]
            image = cv2.imread(os.path.join(self.root_dir, path, img_file))
            mask = cv2.imread(os.path.join(self.root_dir, mask_path, img_file), cv2.IMREAD_UNCHANGED)
            depth = np.load(os.path.join(self.root_dir, depth_path, img_file.replace('png', 'npz')))['data']

            range_4_channels = self._depth2point(depth, camera=depth_path.split('/')[-1])
            image, mask = self._process_image_and_mask(image, mask)
            
            dim = (image.shape[1], image.shape[0]) # Ensure range matches processed image size
            range_4_channels = cv2.resize(range_4_channels, dim, interpolation=cv2.INTER_LINEAR)

            mask = self._remap_mask(mask, threshold=50)

            ims.append(torch.from_numpy(image).float())
            masks.append(torch.from_numpy(mask).long())
            ranges.append(torch.from_numpy(range_4_channels).float())

        ims_tensor = torch.stack(ims).permute(0, 3, 1, 2) / 255.0
        masks_tensor = torch.stack(masks)
        ranges_tensor = torch.stack(ranges).permute(0, 3, 1, 2)

        sample = {'image': ims_tensor, 'mask': masks_tensor, 'range': ranges_tensor}

        if self.transform:
            sample = self.transform(sample)
        elif self.apply_image_transform:
            sample["image"] = self.img_transform(sample["image"])
        
        return sample




class PDDatasetEval(PDDataset3DOD):
    """
    Dataset for evaluation, loading all frames from a sequence.
    This class inherits from PDDataset3DOD to reuse the 3D data handling logic.
    It corresponds to the original `datasetPDEval.py`.
    """
    def __init__(self, root: str, split: str = 'eval'):
        # Hardcode supervision to 'all' as in the original evaluation script
        super().__init__(root, split, supervision='all')

    def _get_split_files(self) -> List[str]:
        """Overrides base method for the specific split logic of this dataset."""
        all_files = sorted(os.listdir(self.root_dir))
        if self.split == 'eval':
            return all_files[0:1]
        elif self.split == 'test':
            return all_files
        return []

    def _setup_paths(self):
        """Sets up paths for evaluation. Masks are always 'ari_masks'."""
        for f in self.files:
            for i in CAMERA_VIEWS:
                camera_path = f'/rgb/camera_0{i}'
                if os.path.exists(os.path.join(self.root_dir, f + camera_path)):
                    self.real_files.append(f + camera_path)
                    self.mask_files.append(f'/ari_masks/camera_0{i}')
                    self.depth_files.append(f'/depth/camera_0{i}')
                    self.calibration_file = os.path.join(self.root_dir, f, 'calibration/f7ac1d5879108bf162666fa9836d1f9ea712a35f.json')

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Loads all 200 frames from a sequence for evaluation."""
        path = self.real_files[index]
        mask_path = self.mask_files[index]
        depth_path = self.depth_files[index]
        all_images = sorted(os.listdir(os.path.join(self.root_dir, path)))

        ims, masks, ranges = [], [], []
        for img_file in all_images:
            image = cv2.imread(os.path.join(self.root_dir, path, img_file))
            mask = cv2.imread(os.path.join(self.root_dir, mask_path, img_file), cv2.IMREAD_UNCHANGED)
            depth = np.load(os.path.join(self.root_dir, depth_path, img_file.replace('png', 'npz')))['data']

            range_4_channels = self._depth2point(depth, camera=depth_path.split('/')[-1])
            image, mask = self._process_image_and_mask(image, mask)
            
            dim = (image.shape[1], image.shape[0])
            range_4_channels = cv2.resize(range_4_channels, dim, interpolation=cv2.INTER_LINEAR)

            # Use a different mask remapping threshold for evaluation
            mask = self._remap_mask(mask, threshold=500)

            # Transform and permute each frame individually
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            
            ims.append(self.img_transform(image_tensor))
            masks.append(torch.from_numpy(mask).long())
            ranges.append(torch.from_numpy(range_4_channels).float().permute(2, 0, 1))

        return {
            'image': torch.stack(ims),
            'mask': torch.stack(masks),
            'range': torch.stack(ranges)
        }
