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
    
    # DIP/physics: normalize by atmospheric light A to estimate haze thickness.
    norm_image = image / atmospheric_light
    
    # DCP step: min over channels, min_c (I_c / A_c).
    min_channel = np.min(norm_image, axis=2)
    
    # Morphological min filter over local patch Omega(x).
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (patch_size, patch_size)
    )
    
    # DCP transmission model: t(x)=1-omega*min_{y in Omega(x)} min_c I_c(y)/A_c.
    transmission = 1 - omega * cv2.erode(min_channel, kernel)
    
    # DIP smoothing: suppress block artifacts/noise from hard min operations.
    transmission = cv2.GaussianBlur(transmission, (15, 15), 0)
    
    # Numerical stability for reconstruction; prevents division blow-up later.
    transmission = np.clip(transmission, 0.1, 1)
    
    return transmission