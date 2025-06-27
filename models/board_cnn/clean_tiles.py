import cv2, os
from tqdm import tqdm
import numpy as np

ROOT = '../../models_data/train/W'      # A/…/Z folders
SIZE = 90                             # final canvas size
BORDER_SIZE = 10

image_path = os.listdir(ROOT)[1]
image_path = os.path.join(ROOT, image_path)
print(image_path)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print(f'shape: {image.shape}')

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# bordered_image = cv2.copyMakeBorder(image, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, value=(0, 0, 0))
bordered_image = np.zeros_like(image)
# [10:80]
bordered_image[BORDER_SIZE:SIZE-BORDER_SIZE][BORDER_SIZE:SIZE-BORDER_SIZE] = image[BORDER_SIZE:SIZE-BORDER_SIZE][BORDER_SIZE:SIZE-BORDER_SIZE]
print(f'shape: {bordered_image.shape}')

# Display the image with the border
cv2.imshow('Image with Black Border', bordered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()