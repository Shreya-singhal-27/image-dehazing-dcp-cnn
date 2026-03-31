import torch
import numpy as np
import cv2
from models.unet import UNet


def refine_transmission_unet(image, transmission):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_weights.pth", map_location=device))
    model.eval()

    h, w = transmission.shape

    # resize to multiple of 32
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32

    image_resized = cv2.resize(image, (new_w, new_h))
    transmission_resized = cv2.resize(transmission, (new_w, new_h))

    image_resized = np.transpose(image_resized, (2, 0, 1))
    transmission_resized = transmission_resized[np.newaxis, :, :]

    input_tensor = np.concatenate([image_resized, transmission_resized], axis=0)
    input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(device)

    with torch.no_grad():
        refined = model(input_tensor).squeeze().cpu().numpy()

    refined = cv2.resize(refined, (w, h))
    refined = np.clip(refined, 0.1, 1)

    return refined