

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def open_image(file_path):
    return Image.open(file_path)

def convert_to_grayscale(image):
    return image.convert("L")

def crop_borders(img, l=0.01, r=0.01, u=0.04, d=0.04):
    nrows, ncols = img.shape
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1 - r))
    u_crop = int(nrows * u)
    d_crop = int(nrows * (1 - d))
    return img[u_crop:d_crop, l_crop:r_crop]

def process_images(data, output_dir, limit=400):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for index, row in data.iterrows():
        if index >= limit:
            break
        mammogram = cv2.imread(row[11], cv2.IMREAD_GRAYSCALE)
        roi = cv2.imread(row[13], cv2.IMREAD_GRAYSCALE)
        roi_resized = cv2.resize(roi, (mammogram.shape[1], mammogram.shape[0]))
        overlay = cv2.addWeighted(mammogram, 0.7, roi_resized, 0.3, 0)
        output_filename = os.path.join(output_dir, f'{row[0]}.png')
        cv2.imwrite(output_filename, overlay)
