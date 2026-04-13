import cv2
import numpy as np

def extract_color_features(roi_image, mask_2d):
    # pastiin bukan background
    valid_mask = mask_2d > 0

    #RGB
    valid_pixels_rgb = roi_image[valid_mask] 
    mean_r, mean_g, mean_b = np.mean(valid_pixels_rgb, axis=0)

    # convert ke CIELab
    roi_lab = cv2.cvtColor(roi_image, cv2.COLOR_RGB2LAB)
    valid_pixels_lab = roi_lab[valid_mask]
    mean_l, mean_a, mean_b = np.mean(valid_pixels_lab, axis=0)

    # convert ke HSV
    roi_hsv = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
    valid_pixels_hsv = roi_hsv[valid_mask]
    mean_h, mean_s, mean_v = np.mean(valid_pixels_hsv, axis=0)

    features = {
        'mean_R': mean_r, 'mean_G': mean_g, 'mean_B': mean_b,
        'mean_L': mean_l, 'mean_a': mean_a, 'mean_b': mean_b,
        'mean_H': mean_h, 'mean_S': mean_s, 'mean_V': mean_v
    }

    return features