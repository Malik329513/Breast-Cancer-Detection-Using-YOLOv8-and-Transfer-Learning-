

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

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    clahe_l = clahe.apply(l)
    clahe_lab = cv2.merge((clahe_l, a, b))
    return cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2BGR)

def haze_reduced_local_global(hazy_image, window_size=15, epsilon=0.001):
    hazy_lab = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2LAB)
    hazy_l, hazy_a, hazy_b = cv2.split(hazy_lab)
    atmospheric_light = np.max(hazy_l)
    hazy_dark = cv2.erode(hazy_l, cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size)))
    transmission_map = 1 - hazy_dark / atmospheric_light
    refined_transmission_map = cv2.max(transmission_map, epsilon)
    inverse_transmission_map = 1 / refined_transmission_map
    dehazed_l = (hazy_l.astype(np.float32) - atmospheric_light) * inverse_transmission_map + atmospheric_light
    dehazed_l = np.clip(dehazed_l, 0, 255).astype(np.uint8)
    dehazed_lab = cv2.merge((dehazed_l, hazy_a, hazy_b))
    return cv2.cvtColor(dehazed_lab, cv2.COLOR_LAB2BGR)

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
