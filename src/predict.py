"""
Single-image prediction with optional Grad-CAM visualization.

Usage:
    python -m src.predict --image-path path/to/nail.png --model-path weights/best_pyramid.pth
    python -m src.predict --image-path path/to/nail.png --model-path weights/best_pyramid.pth --show-gradcam
"""

import argparse
import os
import torch
import cv2
import numpy as np
from PIL import Image

from .transforms import get_val_transforms
from .models.deit_classifier import DeiTClassifier
from .models.pyramid_transformer import ConvPyramidTransformerCBAM
from .utils.preprocessing import apply_clahe


def preprocess_image(img_path: str, image_size: int = 224):
    """Load, enhance with CLAHE, and transform a nail image for inference."""
    image = Image.open(img_path).convert('RGB')
    enhanced = apply_clahe(image)
    transform = get_val_transforms((image_size, image_size))
    tensor = transform(enhanced).unsqueeze(0)
    return tensor, enhanced


def generate_gradcam(model, input_tensor, target_layer, save_path: str):
    """Generate and save a Grad-CAM heatmap overlay."""
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    model.eval()
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Compute Grad-CAM
    grads = gradients['value']
    acts = activations['value']
    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Overlay on original image
    img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (224, 224))

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    handle_fwd.remove()
    handle_bwd.remove()

    print(f"  Grad-CAM saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict Anemia from Fingernail Image")
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the nail image')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='pyramid',
                        choices=['deit', 'pyramid'],
                        help='Model architecture')
    parser.add_argument('--show-gradcam', action='store_true',
                        help='Generate Grad-CAM visualization')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # --- Load Model ---
    if args.model_type == 'deit':
        model = DeiTClassifier(num_classes=2).to(args.device)
    else:
        model = ConvPyramidTransformerCBAM(num_classes=2).to(args.device)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # --- Predict ---
    input_tensor, orig_img = preprocess_image(args.image_path)
    input_tensor = input_tensor.to(args.device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    class_names = ["Non-Anemic", "Anemic"]
    pred_idx = outputs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item() * 100

    print("\n" + "="*40)
    print(f"  Prediction:  {class_names[pred_idx]}")
    print(f"  Confidence:  {confidence:.1f}%")
    print(f"  Non-Anemic:  {probs[0, 0].item() * 100:.1f}%")
    print(f"  Anemic:      {probs[0, 1].item() * 100:.1f}%")
    print("="*40)

    # --- Grad-CAM ---
    if args.show_gradcam:
        os.makedirs('results', exist_ok=True)
        if args.model_type == 'pyramid':
            target_layer = model.encoder[-1]  # Last conv block
        else:
            target_layer = model.backbone.blocks[-1].norm1

        input_tensor.requires_grad_(True)
        generate_gradcam(model, input_tensor, target_layer, 'results/gradcam_output.png')


if __name__ == '__main__':
    main()
