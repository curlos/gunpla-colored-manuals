"""Test: strip grayscale shading from a rendered manual image, keep only line art."""

import cv2
import numpy as np

SOURCE = "/Users/curlos/.claude/image-cache/79e3d696-269d-4fcc-9a61-4d5b8ca39543/5.png"

img = cv2.imread(SOURCE, cv2.IMREAD_GRAYSCALE)

# Canny: classic gradient-based edge detector.
canny = cv2.Canny(img, threshold1=40, threshold2=120)
canny_lineart = 255 - canny  # black lines on white, like the manuals themselves

# XDoG-style: difference of two Gaussian blurs, thresholded — tends to give
# cleaner, more consistent line weight than raw Canny on shaded renders.
blur1 = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8)
blur2 = cv2.GaussianBlur(img, (0, 0), sigmaX=1.6)
dog = blur1.astype(np.float32) - blur2.astype(np.float32)
xdog = np.where(dog < -2, 0, 255).astype(np.uint8)

cv2.imwrite("lineart_canny.png", canny_lineart)
cv2.imwrite("lineart_xdog.png", xdog)
print("Wrote lineart_canny.png and lineart_xdog.png")
