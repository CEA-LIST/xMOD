# train.py

import argparse
import os
import yaml
import torch
from trainer import Trainer
from utils import set_seeds

def get_config():
    """
    Parses command-line arguments and loads configuration from a YAML file.
    This function consolidates settings for both KITTI and PD datasets.
    """
    parser = argparse.ArgumentParser(description="Unified Training Script for Object Discovery")

    # =================================================================================
    # >> Core Configuration <<
    # =================================================================================
    parser.add_argument('--dataset', default='kitti', type=str, choices=['kitti', 'pd'],
                        help='The dataset to use for training (kitti or pd).')
    parser.add_argument('--config', type=str, help='Path to a YAML configuration file to override defaults.')
    parser.add_argument('--mode', default='2d_3d_ts', type=str,
                        help="Training mode: '2d_burn_in', '3d_burn_in', '2d_ts', '3d_ts', '2d_3d_ts'.")

    # =================================================================================
    # >> Paths and Environment <<
    # =================================================================================
    parser.add_argument('--data_path', type=str, help='Path to the dataset root directory.')
    parser.add_argument('--model_dir', default='./tmp/', type=str, help='Directory to save model checkpoints.')
    parser.add_argument('--sample_dir', default='./samples/', type=str, help='Directory to save visualization samples.')
    parser.add_argument('--exp_name', default='unified_exp', type=str, help='A name for the current experiment.')
    parser.add_argument('--checkpoint_path_2d', type=str, help='Path to the 2D model checkpoint for TS training.')
    parser.add_argument('--checkpoint_path_3d', type=str, help='Path to the 3D model checkpoint for TS training.')

    # =================================================================================
    # >> W&B Logging <<
    # =================================================================================
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging.')
    parser.add_argument('--proj_name', default='unified-object-discovery', type=str, help='W&B project name.')
    parser.add_argument('--entity', default='', type=str, help='W&B entity name.')

    # =================================================================================
    # >> Training Hyperparameters <<
    # =================================================================================
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--warmup_steps', default=3000, type=int)
    parser.add_argument('--decay_rate', default=0.5, type=float)
    parser.add_argument('--decay_steps', default=50000, type=int)

    # =================================================================================
    # >> Model Architecture <<
    # =================================================================================
    parser.add_argument('--num_slots', default=45, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--hid_dim', default=64, type=int, help='Hidden dimension size in the model.')

    # =================================================================================
    # >> Loss Function Weights <<
    # =================================================================================
    parser.add_argument('--weight_recon', default=1.0, type=float)
    parser.add_argument('--weight_mask', default=1.0, type=float)
    parser.add_argument('--weight_nll', default=1.0, type=float)
    parser.add_argument('--weight_reg', default=0.3, type=float)

    # =================================================================================
    # >> Teacher-Student (TS) Learning <<
    # =================================================================================
    parser.add_argument('--start_teacher_epoch', default=0, type=int, help="Epoch to start knowledge distillation.")
    parser.add_argument('--teacher_momentum', default=0.996, type=float, help="Momentum for EMA of teacher weights.")
    parser.add_argument('--teacher_confidence_threshold', default=0.9, type=float, help='Confidence threshold for filtering teacher pseudo-labels.')

    # =================================================================================
    # >> Data Augmentation <<
    # =================================================================================
    # Shared
    parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=float, nargs="+")
    parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=float, nargs="+")
    parser.add_argument('--crop_scale', default=[0.75, 1.0], type=float, nargs="+")
    parser.add_argument('--crop_ratio', default=[2.0, 2.0], type=float, nargs="+")
    parser.add_argument('--p_flip', default=0.4, type=float)
    parser.add_argument('--p_crop', default=0.4, type=float)
    # 2D Specific
    parser.add_argument('--brightness', default=0.4, type=float)
    parser.add_argument('--contrast', default=0.4, type=float)
    parser.add_argument('--saturation', default=0.4, type=float)
    parser.add_argument('--hue', default=0.1, type=float)
    parser.add_argument('--p_contrastive', default=0.8, type=float)
    parser.add_argument('--p_grayscale', default=0.2, type=float)
    # 3D Specific
    parser.add_argument('--p_drop', default=0.1, type=float, help="Point drop probability.")
    parser.add_argument('--p_jitter', default=0.4, type=float, help="Point jitter probability.")
    
    # --- Parse known args to load YAML and set dataset defaults ---
    args, remaining_argv = parser.parse_known_args()

    # --- Set Dataset-Specific Defaults ---
    if args.dataset == 'kitti':
        defaults = {
            'data_path': './KITTI_DOM/KITTI_DOM_train/',
            'resolution': (368, 1248),
            'mask_resolution': (92, 312),
            'supervision': 'all' # KITTI uses all masks
        }
    elif args.dataset == 'pd':
        defaults = {
            'data_path': './TRI_PD/TRI_PD_train',
            'resolution': (480, 968),
            'mask_resolution': (120, 242),
            'supervision': 'est' # PD can use different supervision types
        }
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    parser.set_defaults(**defaults)

    # --- Load from YAML if provided ---
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # Map YAML keys to argparse destinations for compatibility
        key_map = {'weight_NLL': 'weight_nll', 'start_teacher': 'start_teacher_epoch'}
        mapped_config = {key_map.get(k, k): v for k, v in yaml_config.items()}
        parser.set_defaults(**mapped_config)

    # --- Final parse with all defaults set ---
    opt = parser.parse_args(remaining_argv)
    return opt


def main():
    """
    Main function to initialize and run the training process.
    """
    opt = get_config()
    print("--- Starting training with the following configuration ---")
    print(opt)
    print("---------------------------------------------------------")

    set_seeds(opt.seed)

    if opt.wandb:
        try:
            import wandb
            wandb.init(project=opt.proj_name, entity=opt.entity, name=opt.exp_name, config=opt)
        except ImportError:
            print("W&B not installed. Please run `pip install wandb` to enable logging.")
            opt.wandb = False

    trainer = Trainer(opt)
    trainer.train()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.autograd.set_detect_anomaly(True)
    main()
