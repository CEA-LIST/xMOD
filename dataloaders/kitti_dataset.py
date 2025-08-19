import os
import random
from typing import Callable, Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import dataloaders.kitti_util as utils


class BaseKITTIDataset(Dataset):
    """
    A base class for KITTI datasets to handle common functionalities like
    initialization of paths, resolution, and image transformations.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        resolution: Tuple[int, int] = (1248, 368),
        transform: Callable = None,
        apply_img_transform: bool = True,
        dino: bool = False
    ):
        """
        Initializes the base dataset.

        Args:
            root (str): The root directory of the dataset.
            split (str): The dataset split, e.g., 'train' or 'eval'.
            resolution (Tuple[int, int]): The target resolution for images (width, height).
            transform (Callable, optional): A function/transform to be applied to the sample.
            apply_img_transform (bool): Whether to apply the default image normalization.
            dino (bool): Flag to use DINO-specific resolutions and transformations.
        """
        super().__init__()
        self.root_dir = root
        self.split = split
        self.dino = dino
        self.transform = transform
        self.apply_image_transform = apply_img_transform

        # Set resolutions based on the dino flag
        if self.dino:
            self.resolution = (980, 490)
            self.dresolution = (242, 120)
        else:
            self.resolution = (resolution[0], resolution[1])
            self.dresolution = (resolution[0] // 4, resolution[1] // 4)

        # Define image transformations
        if self.dino:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.resolution[::-1]), # H, W
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.files = self._setup_files()

    def _setup_files(self) -> List[str]:
        """
        Abstract method to be implemented by subclasses to set up the file lists.
        This method should return a list of file identifiers.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.files)

    def _load_and_preprocess_image(self, path: str, is_mask: bool = False) -> np.ndarray:
        """
        Loads an image or mask and resizes it.

        Args:
            path (str): The path to the image file.
            is_mask (bool): True if the image is a segmentation mask.

        Returns:
            np.ndarray: The loaded and resized image.
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {path}")

        resolution = self.resolution if not is_mask else self.dresolution
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        
        # Note: cv2.resize expects (width, height)
        return cv2.resize(img, resolution, interpolation=interpolation)


class PointCloudMixin:
    """
    A mixin class to handle 3D point cloud processing for KITTI datasets.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the PointCloudMixin."""
        super().__init__(*args, **kwargs)
        # Min/max values for normalizing 4D point cloud coordinates (x, y, z, depth)
        self.mins = np.array([-52, -4, 1, 2])
        self.maxs = np.array([53, 15, 70, 80])

    def _process_point_cloud(self, pc_path: str, calib_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Processes a Velodyne point cloud scan to generate a 4D front projection.

        Args:
            pc_path (str): Path to the .bin point cloud file.
            calib_path (str): Path to the calibration file or directory.
            image_shape (Tuple[int, int]): The shape of the original image (height, width).

        Returns:
            np.ndarray: The processed 4D front projection, resized to the target resolution.
        """
        pc_velo = utils.load_velo_scan(pc_path)
        calib = utils.Calibration(calib_path, from_video=True)
        height, width, _ = image_shape

        # Project points to camera and then to image coordinates
        pts_3d_rect = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pts_2d = calib.project_rect_to_image(pts_3d_rect)

        # Filter points that are within the image frame and in front of the camera
        fov_inds = (pts_2d[:, 0] < width) & (pts_2d[:, 0] >= 0) & \
                   (pts_2d[:, 1] < height) & (pts_2d[:, 1] >= 0) & \
                   (pc_velo[:, 0] > 2.0)

        imgfov_pc_velo = pts_3d_rect[fov_inds, :]
        imgfov_pc_velo[:, 2] = np.clip(imgfov_pc_velo[:, 2], a_min=None, a_max=70)
        imgfov_pts_2d = pts_2d[fov_inds, :]

        # Create 4D coordinates (x, y, z, depth)
        depth = np.linalg.norm(imgfov_pc_velo, 2, axis=1).reshape(-1, 1)
        coordinates_4d = np.hstack((imgfov_pc_velo, depth))

        # Normalize coordinates
        normalized_coords = (coordinates_4d - self.mins) / (self.maxs - self.mins)
        normalized_coords = np.clip(normalized_coords, 1e-3, 1 - 1e-3)

        # Create front projection map
        front_proj = np.zeros((height, width, 4), dtype=np.float32)
        imgfov_pts_2d_int = np.floor(imgfov_pts_2d).astype(int)
        
        # Efficiently map 2D points to their 4D coordinates
        front_proj[imgfov_pts_2d_int[:, 1], imgfov_pts_2d_int[:, 0]] = normalized_coords
        
        # Resize to target resolution
        return cv2.resize(front_proj, self.resolution, interpolation=cv2.INTER_NEAREST)


class KITTIDataset(BaseKITTIDataset):
    """
    Dataset for loading standard KITTI sequences for training/validation.
    """

    def _setup_files(self) -> List[Dict[str, str]]:
        """Sets up the file paths for the dataset sequences."""
        sequence_dirs = sorted(os.listdir(self.root_dir))[:151]
        
        if self.split == 'train':
            sequences = sequence_dirs[5:]
        else: # 'val' or 'test'
            sequences = sequence_dirs[:5]

        file_list = []
        for seq in sequences:
            for subdir in ['image_02', 'image_03']:
                if os.path.exists(os.path.join(self.root_dir, seq, subdir)):
                    file_list.append({
                        "image_dir": os.path.join(self.root_dir, seq, subdir, 'data'),
                        "mask_dir": os.path.join(self.root_dir, seq, subdir, 'raft_seg')
                    })
        return file_list

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a sample from the dataset. A sample consists of a sequence of 5 frames.
        """
        paths = self.files[index]
        all_images = sorted(os.listdir(paths["image_dir"]))
        
        # Select a random starting point for a 5-frame sequence
        start_idx = random.randint(0, len(all_images) - 6)
        frame_indices = range(start_idx, start_idx + 5)

        ims, masks = [], []
        for i in frame_indices:
            img_path = os.path.join(paths["image_dir"], all_images[i])
            mask_path = os.path.join(paths["mask_dir"], all_images[i])

            image = self._load_and_preprocess_image(img_path, is_mask=False)
            mask = self._load_and_preprocess_image(mask_path, is_mask=True)

            ims.append(torch.from_numpy(image))
            masks.append(torch.from_numpy(mask))

        # Stack, normalize, and permute images
        ims = torch.stack(ims).float() / 255.0
        ims = ims.permute(0, 3, 1, 2) # T, C, H, W

        masks = torch.stack(masks).long() # T, H, W

        sample = {'image': ims, 'mask': masks}

        if self.transform:
            sample = self.transform(sample)
        elif self.apply_image_transform:
            sample["image"] = self.img_transform(sample["image"])

        return sample


class KITTI3DDataset(PointCloudMixin, KITTIDataset):
    """
    Dataset for KITTI sequences with 3D point cloud data.
    Inherits from PointCloudMixin for 3D processing and KITTIDataset for sequence loading.
    """

    def _setup_files(self) -> List[Dict[str, str]]:
        """Sets up file paths, including velodyne scans."""
        file_list = super()._setup_files()
        for item in file_list:
            seq_dir = os.path.dirname(os.path.dirname(os.path.dirname(item["image_dir"])))
            item["scan_dir"] = os.path.join(seq_dir, 'velodyne_points', 'data')
            item["calib_dir"] = seq_dir
        return file_list

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a 5-frame sample with images, masks, and 3D range projections.
        """
        paths = self.files[index]
        all_images = sorted(os.listdir(paths["image_dir"]))
        
        start_idx = 5 # Fixed for consistency in 3D data example
        frame_indices = range(start_idx, start_idx + 5)

        ims, masks, s_projs = [], []
        for i in frame_indices:
            img_filename = all_images[i]
            img_path = os.path.join(paths["image_dir"], img_filename)
            mask_path = os.path.join(paths["mask_dir"], img_filename)
            pc_path = os.path.join(paths["scan_dir"], img_filename.replace('.png', '.bin'))

            # Load original image to get its shape for point cloud processing
            original_image = cv2.imread(img_path)
            
            image = cv2.resize(original_image, self.resolution, interpolation=cv2.INTER_LINEAR)
            mask = self._load_and_preprocess_image(mask_path, is_mask=True)
            
            front_proj = self._process_point_cloud(pc_path, paths["calib_dir"], original_image.shape)
            
            # Additional filtering logic from the original script can be added here if needed
            # For example, filtering masks based on depth values in front_proj

            ims.append(torch.from_numpy(image))
            masks.append(torch.from_numpy(mask))
            s_projs.append(torch.from_numpy(front_proj))

        # Process and stack tensors
        ims = torch.stack(ims).float() / 255.0
        ims = ims.permute(0, 3, 1, 2)
        masks = torch.stack(masks).long()
        s_projs = torch.stack(s_projs).float().permute(0, 3, 1, 2)

        sample = {
            'image': ims, 
            'mask': masks, 
            'range': s_projs, 
            'scene': os.path.basename(paths["calib_dir"]),
            'real_idx': list(frame_indices)
        }

        if self.transform:
            sample = self.transform(sample)
        elif self.apply_image_transform:
            sample["image"] = self.img_transform(sample["image"])

        return sample


class KITTIEvalDataset(BaseKITTIDataset):
    """
    Dataset for loading single frames for evaluation.
    """

    def _setup_files(self) -> List[str]:
        """Sets up file paths for evaluation data."""
        rgb_dir = os.path.join(self.root_dir, 'rgb')
        files = sorted(os.listdir(rgb_dir))
        return files[:5] if self.split == 'eval' else files

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Retrieves a single image and mask for evaluation."""
        filename = self.files[index]
        img_path = os.path.join(self.root_dir, 'rgb', filename)
        mask_path = os.path.join(self.root_dir, 'instance', filename)

        image = self._load_and_preprocess_image(img_path, is_mask=False)
        mask = self._load_and_preprocess_image(mask_path, is_mask=True)

        # Convert to tensor, normalize, and permute
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1) # C, H, W
        mask = torch.from_numpy(mask).long() # H, W

        sample = {'image': image, 'mask': mask}

        if self.apply_image_transform:
            sample["image"] = self.img_transform(sample["image"])
        
        return sample


class KITTI3DEvalDataset(PointCloudMixin, BaseKITTIDataset):
    """
    Dataset for loading single frames with 3D data for evaluation.
    """

    def _setup_files(self) -> List[str]:
        """Sets up file paths based on calibration files."""
        calib_dir = os.path.join(self.root_dir, 'calib')
        files = sorted([os.path.splitext(f)[0] for f in os.listdir(calib_dir)])
        return files[:5] if self.split == 'eval' else files

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieves a single sample with image, mask, and 3D data."""
        file_id = self.files[index]
        
        img_path = os.path.join(self.root_dir, 'rgb', f"{file_id}.png")
        mask_path = os.path.join(self.root_dir, 'instance', f"{file_id}.png")
        pc_path = os.path.join(self.root_dir, 'pc', f"{file_id}.bin")
        calib_path = os.path.join(self.root_dir, 'calib', file_id)

        original_image = cv2.imread(img_path)
        image = cv2.resize(original_image, self.resolution, interpolation=cv2.INTER_LINEAR)
        mask = self._load_and_preprocess_image(mask_path, is_mask=True)
        front_proj = self._process_point_cloud(pc_path, calib_path, original_image.shape)

        # Convert to tensor, normalize, and permute
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        front_proj = torch.from_numpy(front_proj).float().permute(2, 0, 1)

        sample = {
            'image': self.img_transform(image) if self.apply_image_transform else image,
            'mask': mask,
            'range': front_proj,
            'pc_velo_path': pc_path,
            'calib_path': calib_path
        }
        
        return sample

