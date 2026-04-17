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
    
    # Keep scale consistent: normalize only if input is in 0..255.
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    
    # DIP (Dark Channel Prior): per-pixel min over RGB, min_c I_c(x).
    min_channel = np.min(image, axis=2)
    
    # DIP (Morphology): local erosion approximates patch-wise minimum,
    # min_{y in Omega(x)} min_c I_c(y), i.e., the dark channel.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, 
        (patch_size, patch_size)
    )
    
    dark_channel = cv2.erode(min_channel, kernel)
    
    return dark_channel