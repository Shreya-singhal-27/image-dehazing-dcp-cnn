import cv2
import numpy as np

from utils.dark_channel import get_dark_channel
from utils.atmospheric_light import estimate_atmospheric_light
from utils.transmission import estimate_transmission
from utils.dehaze import recover_image
from utils.unet_refine import refine_transmission_unet


image = cv2.imread("dataset/hazy/hazy_img1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0

dark = get_dark_channel(image)

A = estimate_atmospheric_light(image, dark)

transmission = estimate_transmission(image, A)

# CNN refinement
refined = refine_transmission_unet(image, transmission)
refined = cv2.GaussianBlur(refined, (7,7), 0)
dehazed = recover_image(image, refined, A)

cv2.imwrite("outputs/refined_transmission.png", (refined * 255).astype("uint8"))
cv2.imwrite("outputs/dehazed.png", (dehazed * 255).astype("uint8"))

print("CNN refined dehazed image saved")