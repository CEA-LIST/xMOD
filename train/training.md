
# Training Procedures

This guide outlines the instructions for running the different training modes with the unified script `train.py`.

While all parameters can be set via command-line arguments, the recommended approach is to use a **`.yaml` configuration file** to define the training parameters. This simplifies the command and makes experiments more reproducible. You can override any parameter in the `.yaml` file by specifying it as a command-line argument.

All training modes are controlled via the **`--mode`** argument.

-----

## Configuration

You can configure a training run in two ways:

1.  **YAML Config File (Recommended)**: Specify a configuration file using the `--config` argument. This is the cleanest way to manage all hyperparameters.

    ```bash
    python train.py --config /path/to/your/config.yaml
    ```

2.  **Command-Line Overrides**: Any argument specified on the command line will **override** the value in the `.yaml` file. This is useful for quickly experimenting with different settings.

    ```bash
    # This will use all settings from the config file but run with a batch size of 16
    python train.py --config /path/to/your/config.yaml --batch_size 16
    ```

-----

## Training Modes

### 1\. 2D Model Burn-in (Supervised Training)

This mode trains the 2D (image-based) model from scratch using ground-truth labels. This is the first step before starting teacher-student learning.

```bash
# All parameters are defined in the config file
python train.py \
    --mode 2d_burn_in \
    --config ./configs/cfg_burnin_2D.yaml
```

### 2\. 3D Model Burn-in (Supervised Training)

Similarly, this mode trains the 3D (point cloud-based) model from scratch using ground-truth labels. This is a prerequisite for 3D teacher-student learning.

```bash
# Use the provided 3D burn-in config file
python train.py \
    --mode 3d_burn_in \
    --config ./configs/cfg_burnin_3D.yaml
```

### 3\. 2D Teacher-Student (TS) Training

This mode refines the 2D model using the teacher-student paradigm. The configuration file should specify the `checkpoint_path_2d` to load the pre-trained model from the burn-in phase.

```bash
# The config file specifies the checkpoint to load
python train.py \
    --mode 2d_ts \
    --config ./configs/cfg_ts_2D.yaml
```

### 4\. 3D Teacher-Student (TS) Training

This mode refines the 3D model using the teacher-student approach. The provided `cgf_ts_3D.yaml` file already contains the necessary settings, including the `path_load` which points to the pre-trained 3D model checkpoint.

```bash
# The config file specifies the checkpoint to load
python train.py \
    --mode 3d_ts \
    --config ./configs/cgf_ts_3D.yaml
```

### 5\. Combined 2D & 3D Teacher-Student (TS) Training

This is the most advanced mode, featuring **cross-modal distillation** where the 2D and 3D models teach each other. The configuration file must specify the paths to both the pre-trained 2D and 3D model checkpoints.

```bash
# The config file specifies both 2D and 3D checkpoints
python train.py \
    --mode 2d_3d_ts \
    --config ./configs/cfg_ts_2D_3D.yaml
```