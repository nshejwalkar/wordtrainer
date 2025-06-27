import cv2, os
from tqdm import tqdm
import numpy as np

ROOT = '../../models_data/train'      # A/…/Z folders
SIZE = 90                             # final canvas size
BORDER_SIZE = 10

# image_path = os.listdir(ROOT)[-3]
# image_path = os.path.join(ROOT, image_path)
# print(image_path)

# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# print(f'shape: {image.shape}')

# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # bordered_image = cv2.copyMakeBorder(image, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, value=(0, 0, 0))
# # bordered_image = np.zeros_like(image)
# # [10:80]
# # bordered_image[BORDER_SIZE:SIZE-BORDER_SIZE][BORDER_SIZE:SIZE-BORDER_SIZE] = image[BORDER_SIZE:SIZE-BORDER_SIZE][BORDER_SIZE:SIZE-BORDER_SIZE]

# blacked = image.copy()

# for i in range(10):
#    for j in range(90):
#       blacked[i][j] = 0
# for i in range(80,90):
#    for j in range(90):
#       blacked[i][j] = 0
# for i in range(90):
#    for j in range(10):
#       blacked[i][j] = 0
#    for j in range(80,90):
#       blacked[i][j] = 0

# print(f'shape: {blacked.shape}')

# # Display the image with the border
# cv2.imshow('Image with Black Border', blacked)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for letter in os.listdir(ROOT):
   if letter.startswith('_'):
      print("skipped", letter)
      continue

   letter = os.path.join(ROOT, letter)
   for tile in os.listdir(letter):
      if not tile.endswith('png'):
         print("not png", tile)
         continue

      tile = os.path.join(letter, tile)
      
      image = cv2.imread(tile, cv2.IMREAD_GRAYSCALE)

      blacked = image.copy()

      for i in range(10):
         for j in range(90):
            blacked[i][j] = 0
      for i in range(80,90):
         for j in range(90):
            blacked[i][j] = 0
      for i in range(90):
         for j in range(10):
            blacked[i][j] = 0
         for j in range(80,90):
            blacked[i][j] = 0
      
      cv2.imwrite(tile, blacked)

      print("overwrote ", tile)