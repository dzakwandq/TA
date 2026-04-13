import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# Menambahkan root directory ke system path agar bisa memanggil folder 'models'
# Ini penting karena file ini berada di dalam folder 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.load_model_TA import build_unbcsm

def load_segmentation_model(weights_path):
    # tinggi 682, lebar 512, 3 Channel RGB
    input_shape = (682, 512, 3) 

    model = build_unbcsm(input_shape)
    
    model.load_weights(weights_path)
    
    print("Model Segmentasi berhasil dimuat!")
    return model

def extract_conjunctiva_roi(model, image):
    """
    - image: format RGB.

    Return:
    - roi_image: hasil konjungtiva (background hitam) ukuran asli resize
    - mask_2d: biner mask konjungtiva
    """
    SIZE_X = 512 # Lebar
    SIZE_Y = 682 # Tinggi

    # resizing
    img_resized = cv2.resize(image, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_LINEAR)
    
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Tambahkan dimensi batch di depan: dari (682, 512, 3) menjadi (1, 682, 512, 3)
    img_input = np.expand_dims(img_normalized, axis=0)

    #prediction
    prediction = model.predict(img_input, verbose=0)

    #thresholding
    thresholded_prediction = (prediction > 0.5).astype(np.uint8)
    
    # Hilangkan dimensi batch dan dimensi channel terakhir: (1, 682, 512, 1) -> (682, 512)
    mask_2d = np.squeeze(thresholded_prediction, axis=(0, 3))
    
    # Aplikasikan mask ke citra asli yang sudah di-resize menggunakan bitwise_and
    # Ini jauh lebih efisien dan aman dari perkalian matriks manual
    roi_image = cv2.bitwise_and(img_resized, img_resized, mask=mask_2d)

    return roi_image, mask_2d