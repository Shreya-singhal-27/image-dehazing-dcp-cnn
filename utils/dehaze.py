import numpy as np


def recover_image(image, transmission, atmospheric_light):
    """
    Recover dehazed image using atmospheric model
    """
    
    # Physics-aware stability floor t0 in J=(I-A)/max(t,t0)+A.
    transmission = np.maximum(transmission, 0.3)
    
    # Expand transmission to 3 channels
    transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
    
    # Atmospheric scattering inversion:
    # I(x)=J(x)t(x)+A(1-t(x))  =>  J(x)=(I(x)-A)/t(x)+A.
    J = (image - atmospheric_light) / transmission + atmospheric_light
    
    # Range clamp and mild gamma to improve visual contrast in output.
    J = np.clip(J, 0, 1)
    J = J ** 0.85
    return J