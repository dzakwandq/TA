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
        # shape: (batch_size, tinggi, lebar, channel)
        X = np.empty((len(batch_x_paths), self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        Y = np.empty((len(batch_y_paths), self.target_size[0], self.target_size[1], 1), dtype=np.float32)

        # 2. Looping memproses setiap gambar di dalam batch
        for i, (img_path, mask_path) in enumerate(zip(batch_x_paths, batch_y_paths)):
            
            # --- BACA DAN RESIZE GAMBAR (Seperti Step 2) ---
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
            
            # --- BACA DAN RESIZE MASK (Seperti Step 1 & 2) ---
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img.shape[2] == 4: # Jika PNG transparan
                _, mask = cv2.threshold(mask_img[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            else:
                gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

            # --- AUGMENTASI (Hanya jika self.augment = True) ---
            if self.augment:
                # Pilih sudut acak antara -45 sampai 45 derajat
                random_angle = random.randint(-45, 45)
                # Pilih kecerahan acak
                random_brightness = random.randint(0, 30)
                
                # ROTASI OTOMATIS: Diterapkan ke 'img' dan 'mask' SEKALIGUS!
                if random_angle != 0:
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), random_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    # Mask ikut dirotasi dengan sudut yang sama persis
                    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                
                # Brightness (hanya untuk gambar input, mask tetap hitam putih)
                if random_brightness != 0:
                    img_float = img.astype(np.float32) + random_brightness
                    np.clip(img_float, 0, 255, out=img_float)
                    img = img_float.astype(np.uint8)

            # --- NORMALISASI (Skala 0.0 - 1.0) ---
            X[i] = img.astype(np.float32) / 255.0
            mask_normalized = mask.astype(np.float32) / 255.0
            Y[i] = np.expand_dims(mask_normalized, axis=-1)

        # Kembalikan sepasang input (X) dan label (Y) yang sudah siap ditraining!
        return X, Y

# =========================================================================
# CARA PENGGUNAAN DI KODE TRAINING UTAMA KAMU:
# =========================================================================
if __name__ == "__main__":
    # Misalkan ini adalah daftar file kamu dari folder dataset
    list_train_images = ['img1.jpg', 'img2.jpg'] # Contoh saja
    list_train_masks = ['mask1.png', 'mask2.png']
    
    list_val_images = ['img3.jpg', 'img4.jpg']
    list_val_masks = ['mask3.png', 'mask4.png']
    
    # 1. Buat Generator untuk Training (Augmentasi = TRUE)
    train_generator = ConjunctivaDataGenerator(
        list_train_images, 
        list_train_masks, 
        batch_size=4, 
        augment=True # <--- Mesin augmentasi menyala
    )
    
    # 2. Buat Generator untuk Validation (Augmentasi = FALSE)
    # Aturan emas: Data validasi/testing TIDAK BOLEH diaugmentasi
    val_generator = ConjunctivaDataGenerator(
        list_val_images, 
        list_val_masks, 
        batch_size=4, 
        augment=False # <--- Mesin augmentasi mati
    )
    
    print("Data generator berhasil disiapkan!")
    
    # Nanti, saat kamu melatih model UNBCSM (seperti di lampiran 2 gambar 4), 
    # kamu tinggal memasukkan generator ini:
    # 
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # history = model.fit(
    #     train_generator,
    #     validation_data=val_generator,
    #     epochs=50
    # )