"""
Visualization utilities for training curves, confusion matrix, ROC curve,
sample predictions, and Grad-CAM heatmaps.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import cv2
import torch


def plot_training_curves(history: dict, save_path: str):
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: File path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss subplot
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax1.set_title('Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy subplot
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
    ax2.set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str):
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.
        save_path: File path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 16})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_scores, save_path: str):
    """Plot and save a ROC curve with AUC.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted probability scores for the positive class.
        save_path: File path to save the figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#0d9488', lw=2.5,
             label=f'Pyramid-CBAM-Transformer (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--',
             label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve — Anemia Detection', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_sample_predictions(model, dataset, n_samples: int, save_path: str,
                            device: str = 'cpu'):
    """Plot a grid of sample predictions from the model.

    Args:
        model: Trained PyTorch model.
        dataset: Dataset to sample from.
        n_samples: Number of samples to display.
        save_path: File path to save the figure.
        device: Device to run inference on.
    """
    model.eval()
    class_names = ["Non-Anemic", "Anemic"]
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_samples > n_cols else [axes]

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            conf = probs[0, pred].item() * 100

        # Denormalize for display
        img = image.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        ax = axes[i]
        ax.imshow(img)
        correct = pred == label.item()
        color = 'green' if correct else 'red'
        symbol = '✓' if correct else '✗'
        ax.set_title(f'{class_names[pred]} {symbol} ({conf:.1f}%)',
                     color=color, fontsize=11, fontweight='bold')
        ax.axis('off')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to {save_path}")
