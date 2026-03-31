import cv2
from utils.dark_channel import get_dark_channel

# Load image
image = cv2.imread("dataset/hazy/hazy_img1.jpg")

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Compute dark channel
dark = get_dark_channel(image)

# Save output
cv2.imwrite("outputs/dark_channel.png", dark * 255)

print("Dark channel saved!")