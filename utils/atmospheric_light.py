import numpy as np


def estimate_atmospheric_light(image, dark_channel):
    """
    Estimate atmospheric light from the hazy image
    
    Args:
        image: RGB image (float [0,1])
        dark_channel: dark channel map
    
    Returns:
        atmospheric light (RGB)
    """
    
    # Flatten dark channel
    flat_dark = dark_channel.ravel()
    
    # Number of pixels
    num_pixels = len(flat_dark)
    
    # Top 0.1% brightest pixels
    num_bright = int(max(num_pixels * 0.001, 1))
    
    # Get indices of brightest pixels
    indices = np.argsort(flat_dark)[-num_bright:]
    
    # Flatten image
    flat_image = image.reshape(num_pixels, 3)
    
    # Atmospheric light = max among brightest pixels
    atmospheric_light = np.max(flat_image[indices], axis=0)
    
    return atmospheric_light