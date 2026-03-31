import numpy as np
import cv2


def estimate_transmission(image, atmospheric_light, omega=0.95, patch_size=15):
    """
    Estimate transmission map
    
    Args:
        image: RGB image [0,1]
        atmospheric_light: atmospheric light vector
        omega: haze amount constant
        patch_size: patch size for min filter
    
    Returns:
        transmission map
    """
    
    # Normalize image by atmospheric light
    norm_image = image / atmospheric_light
    
    # Minimum across channels
    min_channel = np.min(norm_image, axis=2)
    
    # Minimum filter
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (patch_size, patch_size)
    )
    
    transmission = 1 - omega * cv2.erode(min_channel, kernel)
    
    transmission = cv2.GaussianBlur(transmission, (15, 15), 0)
    
    # Avoid very small values
    transmission = np.clip(transmission, 0.1, 1)
    
    return transmission