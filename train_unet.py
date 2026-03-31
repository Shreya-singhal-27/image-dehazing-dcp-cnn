import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from models.unet import UNet

def generate_synthetic():
    # random clean image
    clean = np.random.rand(256,256,3)

    # random transmission
    t = np.random.uniform(0.3,1,(256,256))

    A = np.random.uniform(0.7,1,(1,1,3))

    hazy = clean * t[:,:,None] + A*(1-t[:,:,None])

    return hazy.astype(np.float32), t.astype(np.float32)

device = torch.device("cpu")
model = UNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 3

for epoch in range(epochs):
    total_loss = 0

    for i in range(200):
        hazy, t = generate_synthetic()

        inp = np.transpose(hazy, (2,0,1))
        t = t[np.newaxis,:,:]

        x = np.concatenate([inp, t], axis=0)
        x = torch.from_numpy(x).float().unsqueeze(0)

        y = torch.from_numpy(t).float().unsqueeze(0)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:", epoch, "Loss:", total_loss/200)

torch.save(model.state_dict(), "unet_weights.pth")
print("Training complete. Weights saved.")  