import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils_eval import (
    load_model,
    get_data_loader,
    save_predictions,
)
from metrics import (
    calculate_ari_scores,
    calculate_f1_score,
)
from merging_strategies import merge_predictions


def main(config):
    """
    Main function to run the evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_data_loader(config.test_path, config.batch_size, config.num_workers)

    # Load models based on the evaluation mode
    model_2d = None
    if config.evaluation_mode in ["2d", "2d3d"]:
        model_2d = load_model(
            config.ckpt_path_2d,
            config.resolution,
            config.num_slots,
            config.hid_dim,
            3,
            device,
        )
        model_2d.eval()

    model_3d = None
    if config.evaluation_mode in ["3d", "2d3d"]:
        model_3d = load_model(
            config.ckpt_path_3d,
            config.resolution,
            config.num_slots,
            config.hid_dim,
            4,
            device,
        )
        model_3d.eval()

    # Initialize lists to store metrics
    all_ari_scores, fg_ari_scores = [], []
    all_gt_masks, all_pred_masks, all_scores = [], [], []

    for sample in tqdm(test_loader):
        image = sample["image"].to(device).unsqueeze(1)
        range_image = sample["range"].to(device).unsqueeze(1)
        gt_mask = sample["mask"].unsqueeze(1)

        masks_2d, masks_3d = None, None

        with torch.no_grad():
            if model_2d:
                _, masks_2d, _, _ = model_2d(image)
            if model_3d:
                _, masks_3d, _, _ = model_3d(range_image)

        for i in range(image.size(0)):
            pred_masks = None
            if config.evaluation_mode == "2d":
                pred_masks = masks_2d[i]
            elif config.evaluation_mode == "3d":
                pred_masks = masks_3d[i]
            elif config.evaluation_mode == "2d3d":
                pred_masks = merge_predictions(masks_2d[i], masks_3d[i])

            if config.save_predictions:
                save_predictions(pred_masks, sample["path"][i])

            # Calculate metrics
            if "ari" in config.metrics:
                fg_ari, all_ari = calculate_ari_scores(
                    gt_mask[i], pred_masks, config.num_slots
                )
                if fg_ari is not None and all_ari is not None:
                    fg_ari_scores.append(fg_ari)
                    all_ari_scores.append(all_ari)

            if "f1" in config.metrics:
                # Prepare data for F1 score calculation
                gt_one_hot = F.one_hot(gt_mask[i, 0, :, :])
                pred_one_hot = F.one_hot(
                    pred_masks.argmax(dim=0, keepdim=False), num_classes=config.num_slots
                )

                active_slots = [
                    pred_one_hot[:, :, s]
                    for s in range(config.num_slots - 1)
                    if pred_one_hot[:, :, s].max().item() > 0
                ]
                scores = [1 for _ in range(len(active_slots))]
                gt_slots = [
                    gt_one_hot[:, :, s]
                    for s in range(1, gt_one_hot.shape[2])
                    if gt_one_hot[:, :, s].max().item() > 0
                ]

                all_pred_masks.append(active_slots)
                all_scores.append(scores)
                all_gt_masks.append(gt_slots)

    # Print results
    if "ari" in config.metrics:
        print(f"FG-ARI: {torch.tensor(fg_ari_scores).mean().item():.4f}")
        print(f"All-ARI: {torch.tensor(all_ari_scores).mean().item():.4f}")

    if "f1" in config.metrics:
        f1, precision, recall = calculate_f1_score(
            all_gt_masks, all_scores, all_pred_masks
        )
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument(
        "--evaluation_mode",
        type=str,
        required=True,
        choices=["2d", "3d", "2d3d"],
        help="Evaluation mode: 2d, 3d, or 2d3d",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["ari", "f1"],
        choices=["ari", "f1"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "--ckpt_path_2d",
        type=str,
        default="./tmp/trainWaymo_ts_122/epoch_100.ckpt",
        help="Path to the 2D model checkpoint",
    )
    parser.add_argument(
        "--ckpt_path_3d",
        type=str,
        default="./tmp/trainWaymo_ts_122/epoch_100.ckpt",
        help="Path to the 3D model checkpoint",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="./Waymo_DOM/Waymo_DOM_test",
        help="Path to the test dataset",
    )
    parser.add_argument(
        "--num_slots",
        type=int,
        default=45,
        help="Number of slots in Slot Attention",
    )
    parser.add_argument(
        "--hid_dim", type=int, default=64, help="Hidden dimension size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for the data loader",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save model predictions to disk",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[368, 1248],
        help="Image resolution for the model",
    )
    config = parser.parse_args()

    main(config)