import cv2
import numpy as np

from utils.dark_channel import get_dark_channel
from utils.atmospheric_light import estimate_atmospheric_light
from utils.transmission import estimate_transmission
from utils.dehaze import recover_image
from utils.unet_refine import refine_transmission_unet


# ---------------- Load Image ----------------
image = cv2.imread("dataset/hazy/hazy_img1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# DIP preprocessing: normalize to [0,1] for physical-model computations.
image = image.astype(np.float32) / 255.0


# ---------------- Dark Channel ----------------
# DIP prior: computes local patch-wise dark response for haze cue extraction.
dark = get_dark_channel(image)


# ---------------- Atmospheric Light ----------------
# Estimate global airlight A from haze-dominant bright pixels in dark channel.
A = estimate_atmospheric_light(image, dark)


# ---------------- Transmission ----------------
# Coarse transmission t(x) from DCP equation and local minimum filtering.
transmission = estimate_transmission(image, A)


# ---------------- UNet Refinement ----------------
# Learning-based refinement: fixes halo/block artifacts from coarse DCP t(x).
refined = refine_transmission_unet(image, transmission)
# Extra DIP smoothing for spatial continuity in transmission.
refined = cv2.GaussianBlur(refined, (7, 7), 0)


# ---------------- Recover Image ----------------
# Physical inversion of haze model to recover scene radiance J(x).
dehazed = recover_image(image, refined, A)


# ---------------- Save Outputs ----------------
# ---------------- Save Outputs ----------------
cv2.imwrite("outputs/input.png", (image * 255).astype(np.uint8))

dark_vis = cv2.normalize(dark, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("outputs/dark_channel.png", dark_vis.astype(np.uint8))

trans_vis = cv2.normalize(transmission, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("outputs/transmission.png", trans_vis.astype(np.uint8))

ref_vis = cv2.normalize(refined, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("outputs/refined_transmission.png", ref_vis.astype(np.uint8))

cv2.imwrite("outputs/dehazed.png", (dehazed * 255).astype(np.uint8))


# ---------------- Comparison Image ----------------
comparison = np.hstack([
    (image * 255).astype(np.uint8),
    cv2.cvtColor(dark_vis.astype(np.uint8), cv2.COLOR_GRAY2RGB),
    cv2.cvtColor(trans_vis.astype(np.uint8), cv2.COLOR_GRAY2RGB),
    cv2.cvtColor(ref_vis.astype(np.uint8), cv2.COLOR_GRAY2RGB),
    (dehazed * 255).astype(np.uint8)
])

cv2.imwrite("outputs/comparison.png", comparison)


print("Pipeline complete. Results saved in outputs/")