"""
Unified training script for Anemia Detection models.

Usage:
    python -m src.train --data-dir dataset/Fingernails --model pyramid --epochs 30
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

from .dataset import NailDataset
from .transforms import get_train_transforms, get_val_transforms
from .models.deit_classifier import DeiTClassifier
from .models.pyramid_transformer import ConvPyramidTransformerCBAM
from .utils.visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Train Anemia Detection Model")
    parser.add_argument('--model', type=str, default='pyramid',
                        choices=['deit', 'pyramid'],
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='weights',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print(f"Using device: {args.device}")
    print(f"Model: {args.model}")

    # --- Load Dataset ---
    train_transform = get_train_transforms((args.image_size, args.image_size))
    val_transform = get_val_transforms((args.image_size, args.image_size))

    full_dataset = NailDataset(args.data_dir, transform=train_transform)
    dist = full_dataset.get_class_distribution()
    print(f"Dataset: {len(full_dataset)} images | Anemic: {dist[1]} | Non-Anemic: {dist[0]}")

    # --- Train/Val Split ---
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    # --- Weighted Sampler for Class Imbalance ---
    class_weights = full_dataset.get_class_weights()
    sample_weights = [class_weights[full_dataset.labels[idx]] for idx in train_dataset.indices]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # --- Initialize Model ---
    if args.model == 'deit':
        model = DeiTClassifier(num_classes=2).to(args.device)
    elif args.model == 'pyramid':
        model = ConvPyramidTransformerCBAM(num_classes=2).to(args.device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(args.device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 5
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for inputs, targets in progress:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress.set_postfix({
                'loss': f'{train_loss / (progress.n + 1):.4f}',
                'acc': f'{100. * correct / total:.1f}%'
            })

        train_loss /= len(train_loader)
        train_acc = 100. * correct / total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # --- Checkpointing & Early Stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = os.path.join(args.output_dir, f'best_{args.model}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✅ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1} (patience={patience})")
                break

    # --- Save Results ---
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, 'results/training_curves.png')

    print("\n" + "="*60)
    print("FINAL CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_targets, all_preds, target_names=["Non-Anemic", "Anemic"]))
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
