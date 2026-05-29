import os
import cv2
import numpy as np
import tensorflow as tf
import random

class ConjunctivaDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=8, target_size=(512, 682), augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_paths = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y_paths = self.mask_paths[idx * self.batch_size : (idx + 1) * self.batch_size]

        # array kosong untuk simpan batch
        X = np.empty((len(batch_x_paths), self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        Y = np.empty((len(batch_y_paths), self.target_size[0], self.target_size[1], 1), dtype=np.float32)

        for i, (img_path, mask_path) in enumerate(zip(batch_x_paths, batch_y_paths)):
            
            # --- BACA DAN RESIZE GAMBAR ---
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
            
            # --- LOGIKA EKSTRAKSI MASK HIBRIDA (INDIA & ITALY) ---
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            if mask_img.dtype == np.uint16:
                mask_img = (mask_img / 256).astype(np.uint8)
                
            if len(mask_img.shape) == 3 and mask_img.shape[2] == 4:
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2RGBA)
                
            else :
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            
            mask_hsv = cv2.cvtColor(mask_img, cv2.COLOR_RGB2HSV)
    
            saturation = mask_hsv[:, :, 1]
            
            if np.min(mask_img) == 0:
                _, mask = cv2.threshold(mask_img[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            
            else:    
                _, mask = cv2.threshold(saturation, 1, 255, cv2.THRESH_BINARY)

            kernel = np.ones((6,6), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Resize mask setelah diekstrak
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

            # --- AUGMENTASI ---
            if self.augment:
                #flipping
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                    mask = cv2.flip(mask, 1)

                #rotasi
                random_angle = random.randint(-45, 45)
                if random_angle != 0:
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), random_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                
                #brightness addition
                random_brightness = random.randint(-30, 30)
                if random_brightness != 0:
                    img_float = img.astype(np.float32) + random_brightness
                    np.clip(img_float, 0, 255, out=img_float)
                    img = img_float.astype(np.uint8)

            # --- NORMALISASI ---
            X[i] = img.astype(np.float32) / 255.0
            mask_normalized = mask.astype(np.float32) / 255.0
            Y[i] = np.expand_dims(mask_normalized, axis=-1)

        return X, Y