# trainer.py

import datetime
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# Assuming these are your custom modules
# These will be dynamically selected based on the dataset
from losses import compute_loss
from models.model_bg import SlotAttentionAutoEncoder
from transforms import get_2d_transforms, get_3d_transforms
from utils import (
    update_teacher_model,
    process_teacher_predictions,
    load_model,
    save_checkpoint,
    visualize_kitti,
    visualize_pd,
    evaluate_ari,
)

class Trainer:
    """
    Encapsulates the entire training, evaluation, and checkpointing logic
    in a dataset-agnostic manner.
    """
    def __init__(self, opt):
        """
        Initializes the trainer, setting up models, data loaders, optimizer, and logging.
        """
        self.opt = opt
        self.step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = os.path.join(opt.model_dir, opt.exp_name)
        self.sample_path = os.path.join(opt.sample_dir, opt.exp_name)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        self._init_components()

    def _init_components(self):
        """Initializes models, data loaders, and optimizer based on the config."""
        self.models = {'student': {}, 'teacher': {}}
        params_to_optimize = []

        if '2d' in self.opt.mode:
            self.models['student']['2d'] = self._create_model(output_channels=3)
            params_to_optimize.extend(self.models['student']['2d'].parameters())
            if 'ts' in self.opt.mode:
                self.models['teacher']['2d'] = self._create_model(output_channels=3)

        if '3d' in self.opt.mode:
            self.models['student']['3d'] = self._create_model(output_channels=4)
            params_to_optimize.extend(self.models['student']['3d'].parameters())
            if 'ts' in self.opt.mode:
                self.models['teacher']['3d'] = self._create_model(output_channels=4)

        self._init_dataloaders()
        self.optimizer = optim.Adam(params_to_optimize, lr=self.opt.learning_rate)

    def _create_model(self, output_channels):
        """Factory method to create a model instance."""
        model = SlotAttentionAutoEncoder(
            self.opt.resolution, self.opt.num_slots, self.opt.hid_dim, output_channel=output_channels
        )
        return nn.DataParallel(model).to(self.device)

    def _init_dataloaders(self):
        """
        Initializes datasets and dataloaders based on the chosen dataset,
        using the correct classes and arguments for training and evaluation.
        """
        # 1. Import correct dataset classes based on the dataset argument
        if self.opt.dataset == 'kitti':
            # KITTI has separate classes for training and evaluation
            from dataloaders.kitti_dataset import KITTIDataset, KITTI3DDataset, KITTIEvalDataset, KITTI3DEvalDataset
            TrainDataset2D, TrainDataset3D = KITTIDataset, KITTI3DDataset
            EvalDataset2D, EvalDataset3D = KITTIEvalDataset, KITTI3DEvalDataset
        elif self.opt.dataset == 'pd':
            # PD dataset has different classes for 2D, 3D, and evaluation
            from dataloaders.PD_dataset import PDDataset, PDDataset3DOD, PDDatasetEval
            TrainDataset2D = PDDataset
            TrainDataset3D = PDDataset3DOD
            EvalDataset2D = PDDatasetEval
            EvalDataset3D = PDDatasetEval  # PDDatasetEval handles both modalities
        else:
            raise ValueError(f"Unsupported dataset: {self.opt.dataset}")

        # 2. Instantiate datasets based on the training mode (2D or 3D)
        if '3d' in self.opt.mode:
            train_transform_3d, self.student_transform_3d = get_3d_transforms(self.opt)
            if self.opt.dataset == 'kitti':
                self.train_set = TrainDataset3D(split='train', root=self.opt.data_path, transform=train_transform_3d, resolution=self.opt.resolution)
                self.test_set = EvalDataset3D(split='eval', root=self.opt.data_path, resolution=self.opt.resolution)
            elif self.opt.dataset == 'pd':
                self.train_set = TrainDataset3D(split='train', root=self.opt.data_path, transform=train_transform_3d, supervision=self.opt.supervision)
                self.test_set = EvalDataset3D(split='eval', root=self.opt.data_path)
        else:  # 2D mode
            train_transform_2d, self.student_transform_2d, self.teacher_transform_2d = get_2d_transforms(self.opt)
            if self.opt.dataset == 'kitti':
                self.train_set = TrainDataset2D(split='train', root=self.opt.data_path, transform=train_transform_2d, resolution=self.opt.resolution)
                self.test_set = EvalDataset2D(split='eval', root=self.opt.data_path, resolution=self.opt.resolution)
            elif self.opt.dataset == 'pd':
                self.train_set = TrainDataset2D(split='train', root=self.opt.data_path, transform=train_transform_2d, supervision=self.opt.supervision)
                self.test_set = EvalDataset2D(split='eval', root=self.opt.data_path)

        # 3. Create the DataLoader for the training set
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, drop_last=True, pin_memory=True
        )

    def _load_teacher_checkpoints(self):
        """Loads checkpoints for teacher models at the start of TS training."""
        print("--- Loading Teacher Model Checkpoints ---")
        if '2d' in self.opt.mode and self.opt.checkpoint_path_2d:
            state_dict = torch.load(self.opt.checkpoint_path_2d, map_location=self.device)['model_state_dict']
            load_model(self.models['student']['2d'], state_dict)
            load_model(self.models['teacher']['2d'], state_dict)
        if '3d' in self.opt.mode and self.opt.checkpoint_path_3d:
            state_dict = torch.load(self.opt.checkpoint_path_3d, map_location=self.device)['model_state_dict']
            load_model(self.models['student']['3d'], state_dict)
            load_model(self.models['teacher']['3d'], state_dict)
        print("-----------------------------------------")

    def train(self):
        """Main training loop over epochs."""
        start_time = time.time()
        for epoch in range(self.opt.num_epochs):
            if epoch == self.opt.start_teacher_epoch and 'ts' in self.opt.mode:
                self._load_teacher_checkpoints()

            for model in self.models['student'].values():
                model.train()

            epoch_losses = self._train_epoch(epoch)
            
            log_message = f"Epoch: {epoch + 1}/{self.opt.num_epochs}, Time: {datetime.timedelta(seconds=time.time() - start_time)}"
            for name, value in epoch_losses.items():
                log_message += f", {name}: {value:.4f}"
            print(log_message)

            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.opt.num_epochs:
                self._evaluate_and_save(epoch)
        print("--- Training Finished ---")

    def _train_epoch(self, epoch):
        """Trains the model for a single epoch."""
        total_losses = OrderedDict()
        for sample in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}"):
            self.step += 1
            self._update_learning_rate()

            batch = self._prepare_batch(sample)
            outputs = self._forward_pass(batch, epoch)
            losses = compute_loss(self.opt, batch, outputs, epoch, self.device)

            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            for model in self.models['student'].values():
                clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()

            if 'ts' in self.opt.mode and epoch >= self.opt.start_teacher_epoch:
                if '2d' in self.models['student']:
                    update_teacher_model(self.models['student']['2d'], self.models['teacher']['2d'], self.opt.teacher_momentum)
                if '3d' in self.models['student']:
                    update_teacher_model(self.models['student']['3d'], self.models['teacher']['3d'], self.opt.teacher_momentum)

            for k, v in losses.items():
                total_losses.setdefault(k, 0.0)
                total_losses[k] += v.item()

        return {k: v / len(self.train_dataloader) for k, v in total_losses.items()}

    def _prepare_batch(self, sample):
        """Prepares a batch of data by moving it to the device and applying transforms."""
        batch = {}
        mask_gt = sample['mask'].to(self.device, non_blocking=True)
        interpolated_mask = nn.functional.interpolate(mask_gt.float(), self.opt.mask_resolution).long()

        if '2d' in self.opt.mode:
            image = sample['image'].to(self.device, non_blocking=True)
            batch['student_image_2d'] = self.student_transform_2d(image)
            batch['teacher_image_2d'] = self.teacher_transform_2d(image)
            batch['mask_gt_2d'] = interpolated_mask
        
        if '3d' in self.opt.mode:
            # For PD dataset, 'range_orig' might not exist, so we handle it gracefully.
            range_orig = sample.get('range_orig', sample['range']).to(self.device, non_blocking=True)
            batch['student_range_3d'] = sample['range'].to(self.device, non_blocking=True)
            batch['teacher_range_3d'] = range_orig
            batch['mask_gt_3d'] = interpolated_mask
        return batch

    def _forward_pass(self, batch, epoch):
        """Performs a forward pass for all active models."""
        outputs = {}

        if '2d' in self.models['student']:
            recon, masks, fg, _ = self.models['student']['2d'](batch['student_image_2d'])
            outputs.update({'recon_2d': recon, 'masks_2d': masks, 'mask_fg_2d': fg})

        if '3d' in self.models['student']:
            recon, masks, fg, _ = self.models['student']['3d'](batch['student_range_3d'])
            outputs.update({'recon_3d': recon, 'masks_3d': masks, 'mask_fg_3d': fg})

        if 'ts' in self.opt.mode and epoch >= self.opt.start_teacher_epoch:
            with torch.no_grad():
                if '2d' in self.models['teacher']:
                    _, masks_t, fg_t, _ = self.models['teacher']['2d'](batch['teacher_image_2d'])
                    proc_masks, n_obj = process_teacher_predictions(self.opt, masks_t, fg_t)
                    outputs.update({'teacher_masks_2d': proc_masks, 'teacher_mask_fg_2d': fg_t, 'teacher_n_objects_2d': n_obj})
                if '3d' in self.models['teacher']:
                    _, masks_t, fg_t, _ = self.models['teacher']['3d'](batch['teacher_range_3d'])
                    proc_masks, n_obj = process_teacher_predictions(self.opt, masks_t, fg_t)
                    outputs.update({'teacher_masks_3d': proc_masks, 'teacher_mask_fg_3d': fg_t, 'teacher_n_objects_3d': n_obj})
        return outputs

    def _update_learning_rate(self):
        """Updates the learning rate based on a warmup and decay schedule."""
        if self.step < self.opt.warmup_steps:
            lr = self.opt.learning_rate * (self.step / self.opt.warmup_steps)
        else:
            lr = self.opt.learning_rate * (self.opt.decay_rate ** (self.step / self.opt.decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _evaluate_and_save(self, epoch):
        """Runs evaluation, saves checkpoints, and generates visualizations."""
        print(f"--- Evaluating and Saving at Epoch {epoch + 1} ---")
        
        # Determine the visualization function based on the dataset
        visualize_fn = visualize_kitti if self.opt.dataset == 'kitti' else visualize_pd
        
        if '2d' in self.models['student']:
            model = self.models['student']['2d']
            model.eval()
            visualize_fn(epoch, model, self.test_set, self.opt, self.sample_path, '2d', self.device)
            save_checkpoint(epoch, model, os.path.join(self.model_path, f'epoch_{epoch}_2d.ckpt'))
            
        if '3d' in self.models['student']:
            model = self.models['student']['3d']
            model.eval()
            visualize_fn(epoch, model, self.test_set, self.opt, self.sample_path, '3d', self.device)
            save_checkpoint(epoch, model, os.path.join(self.model_path, f'epoch_{epoch}_3d.ckpt'))

        # Run ARI evaluation only for the PD dataset, as it's specific to it
        if self.opt.dataset == 'pd':
            model_to_eval = self.models['student'].get('2d') or self.models['student'].get('3d')
            if model_to_eval:
                evaluate_ari(model_to_eval, self.test_set, self.device, self.opt.num_slots)
