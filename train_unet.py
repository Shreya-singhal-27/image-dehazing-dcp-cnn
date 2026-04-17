import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from tqdm import tqdm

from models.unet import UNet


# ----------- Haze Generator -----------
def add_haze(image):
    h, w, _ = image.shape

    # DIP/physics proxy: synthetic depth map d(x) via horizontal gradient.
    depth = np.tile(np.linspace(0.1, 1, w), (h, 1))

    # Beer-Lambert law: t(x)=exp(-beta*d(x)) controls transmission decay.
    beta = np.random.uniform(0.6, 1.8)
    transmission = np.exp(-beta * depth)

    # Global atmospheric light A sampled per RGB channel.
    A = np.random.uniform(0.7, 1, (1, 1, 3))

    # Haze image synthesis: I(x)=J(x)t(x)+A(1-t(x)).
    hazy = image * transmission[:, :, None] + A * (1 - transmission[:, :, None])

    return hazy.astype(np.float32), transmission.astype(np.float32)


# ----------- Setup -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

clean_folder = "dataset/clean"

epochs = 10


# ----------- Training -----------
for epoch in range(epochs):
    total_loss = 0
    files = os.listdir(clean_folder)

    for file in tqdm(files):

        path = os.path.join(clean_folder, file)

        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0

        hazy, transmission = add_haze(img)

        # Model input = RGB hazy image + coarse/target transmission channel (4-channel tensor).
        hazy = np.transpose(hazy, (2, 0, 1))
        transmission = transmission[np.newaxis, :, :]

        x = np.concatenate([hazy, transmission], axis=0)

        x = torch.from_numpy(x).float().unsqueeze(0).to(device)
        y = torch.from_numpy(transmission).float().unsqueeze(0).to(device)

        pred = model(x)

        # L1/MAE on transmission encourages edge-preserving refinement.
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(files):.6f}")


# ----------- Save weights -----------
torch.save(model.state_dict(), "unet_weights.pth")
print("Training complete. Weights saved as unet_weights.pth")