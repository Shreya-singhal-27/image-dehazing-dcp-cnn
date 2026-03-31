import torch
import numpy as np
from models.cnn_refinement import TransmissionRefinementCNN


def refine_transmission(image, transmission):

    device = torch.device("cpu")

    model = TransmissionRefinementCNN().to(device)
    model.eval()

    # prepare input
    image = np.transpose(image, (2, 0, 1))
    transmission = transmission[np.newaxis, :, :]

    input_tensor = np.concatenate([image, transmission], axis=0)
    input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0)

    with torch.no_grad():
        refined = model(input_tensor).squeeze().cpu().numpy()

    refined = np.clip(refined, 0.1, 1)

    return refined