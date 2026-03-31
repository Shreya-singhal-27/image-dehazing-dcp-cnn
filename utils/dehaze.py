import numpy as np


def recover_image(image, transmission, atmospheric_light):
    """
    Recover dehazed image using atmospheric model
    """
    
    # Avoid division by zero
    transmission = np.maximum(transmission, 0.3)
    
    # Expand transmission to 3 channels
    transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
    
    # Recover image
    J = (image - atmospheric_light) / transmission + atmospheric_light
    
    # Clip values
    J = np.clip(J, 0, 1)
    J = J ** 0.85
    return J