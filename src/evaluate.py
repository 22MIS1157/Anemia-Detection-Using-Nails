"""
Model evaluation script for Anemia Detection.

Usage:
    python -m src.evaluate --model-path weights/best_pyramid.pth --data-dir dataset/Fingernails
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score

from .dataset import NailDataset
from .transforms import get_val_transforms
from .models.deit_classifier import DeiTClassifier
from .models.pyramid_transformer import ConvPyramidTransformerCBAM
from .utils.visualization import plot_confusion_matrix, plot_roc_curve


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anemia Detection Model")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='pyramid',
                        choices=['deit', 'pyramid'],
                        help='Model architecture type')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Data ---
    val_transform = get_val_transforms((args.image_size, args.image_size))
    val_dataset = NailDataset(args.data_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"Evaluating on {len(val_dataset)} images")

    # --- Load Model ---
    if args.model_type == 'deit':
        model = DeiTClassifier(num_classes=2).to(args.device)
    else:
        model = ConvPyramidTransformerCBAM(num_classes=2).to(args.device)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # --- Run Evaluation ---
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_targets.extend(targets.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # --- Compute Metrics ---
    accuracy = (all_preds == all_targets).mean() * 100
    auroc = roc_auc_score(all_targets, all_probs)
    sensitivity = np.sum((all_preds == 1) & (all_targets == 1)) / np.sum(all_targets == 1) * 100
    specificity = np.sum((all_preds == 0) & (all_targets == 0)) / np.sum(all_targets == 0) * 100

    report = classification_report(all_targets, all_preds,
                                   target_names=["Non-Anemic", "Anemic"])

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(report)
    print(f"Accuracy:    {accuracy:.2f}%")
    print(f"AUROC:       {auroc:.4f}")
    print(f"Sensitivity: {sensitivity:.2f}%")
    print(f"Specificity: {specificity:.2f}%")

    # --- Save Results ---
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
        f.write(f"\nAccuracy:    {accuracy:.2f}%\n")
        f.write(f"AUROC:       {auroc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.2f}%\n")
        f.write(f"Specificity: {specificity:.2f}%\n")

    # --- Generate Visualizations ---
    plot_confusion_matrix(all_targets, all_preds, ["Non-Anemic", "Anemic"],
                          os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_roc_curve(all_targets, all_probs,
                   os.path.join(args.output_dir, 'roc_curve.png'))

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
