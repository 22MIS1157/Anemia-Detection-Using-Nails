"""
Image preprocessing utilities for the Anemia Detection pipeline.

Includes CLAHE enhancement, erythema index computation,
and nail region detection via color thresholding.
"""

import cv2
import numpy as np
from PIL import Image


def apply_clahe(image, clip_limit: float = 3.0, grid_size: tuple = (8, 8)):
    """Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).

    Converts the image to LAB color space, applies CLAHE to the L-channel
    for illumination normalization, then converts back to RGB.

    Args:
        image: PIL Image or numpy array (RGB).
        clip_limit: Contrast limiting threshold for CLAHE.
        grid_size: Size of the grid for histogram equalization.

    Returns:
        PIL Image with enhanced contrast.
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_l = clahe.apply(l_channel)

    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced_rgb)


def compute_erythema_index(image) -> float:
    """Compute the erythema (redness) index from a nail image.

    Uses the a* channel in CIE-LAB color space, which correlates with
    redness/greenness. Lower values indicate pallor (potential anemia).

    Args:
        image: PIL Image or numpy array (RGB).

    Returns:
        Float value representing the mean erythema index.
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    _, a_channel, _ = cv2.split(lab)

    # a* channel: positive = red, negative = green
    # Higher values indicate more redness (healthier nail bed)
    erythema_index = float(np.mean(a_channel))
    return erythema_index


def detect_nail_region(image):
    """Detect and segment the nail region using color thresholding in HSV space.

    Args:
        image: PIL Image or numpy array (RGB).

    Returns:
        Numpy array (RGB) with non-nail regions masked to black.
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Skin/nail tone bounds in HSV
    lower_bound = np.array([0, 20, 70], dtype=np.uint8)
    upper_bound = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask
    segmented = cv2.bitwise_and(img_array, img_array, mask=mask)
    return segmented
