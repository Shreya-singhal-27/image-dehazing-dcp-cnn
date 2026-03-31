import cv2
import numpy as np


def get_dark_channel(image, patch_size=15):
    """
    Compute Dark Channel Prior of an RGB image
    
    Args:
        image: Input RGB image (numpy array)
        patch_size: size of local patch
    
    Returns:
        dark_channel: grayscale dark channel image
    """
    
    # Convert to float [0,1]
    image = image.astype(np.float32) / 255.0
    
    # Take minimum among RGB channels
    min_channel = np.min(image, axis=2)
    
    # Apply minimum filter (erosion)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, 
        (patch_size, patch_size)
    )
    
    dark_channel = cv2.erode(min_channel, kernel)
    
    return dark_channel