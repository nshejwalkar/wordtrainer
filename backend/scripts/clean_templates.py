import cv2
import os
import numpy as np

INPUT_DIR = '../templates'     # folder with your A–Z templates
OUTPUT_DIR = '../templates/clean'  # where to save cleaned ones
TARGET_SIZE = 40  # you can adjust this to match the board tile size

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith('.png'):
        continue

    path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Threshold to binary
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours of the white content
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop to bounding box
    cropped = img[y:y+h, x:x+w]

    # Resize while maintaining aspect ratio
    scale = TARGET_SIZE / max(w, h)
    resized = cv2.resize(cropped, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Pad to square
    pad_x = (TARGET_SIZE - resized.shape[1]) // 2
    pad_y = (TARGET_SIZE - resized.shape[0]) // 2
    padded = cv2.copyMakeBorder(resized, pad_y, TARGET_SIZE - resized.shape[0] - pad_y,
                                pad_x, TARGET_SIZE - resized.shape[1] - pad_x,
                                cv2.BORDER_CONSTANT, value=0)

    # Save
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, padded)
    print(f"Processed {filename}")
