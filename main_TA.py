import cv2
import matplotlib.pyplot as plt
import os

from scripts.segmentation import load_segmentation_model, extract_conjunctiva_roi

from scripts.feature_extraction import extract_color_features

from scripts.hb_predict import load_hb_model, predict_hemoglobin

def main():
    weights_path = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\models\weights\simulasi\model_simulasi_terbaik.keras"
    image_path = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\dataset\Dataset Eyedefy\India\8\20200124_202947.jpg"
    hb_model_path = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\models\weights\hb_model_normal_only.pkl"

    # ini disesuain
    patient_age = 32
    patient_gender = 0  # 1 untuk Laki-laki (M), 0 untuk Perempuan (F)

    # Validasi file
    if not os.path.exists(weights_path) or not os.path.exists(image_path):
        print("File weight atau gambar tidak ditemukan")
        return
    
    # ==========================================
    # 2. INISIALISASI MODEL
    # ==========================================
    segmentation_model = load_segmentation_model(weights_path) #inisiasi modelnya
    hb_model = load_hb_model(hb_model_path) #inisiasi model untuk prediksi HB

    # ==========================================
    # 3. EKSEKUSI PIPELINE
    # ==========================================
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    roi_image, mask_2d = extract_conjunctiva_roi(segmentation_model, img_rgb)

    color_features = extract_color_features(roi_image, mask_2d)

    hb_prediction = predict_hemoglobin(hb_model, color_features, patient_age, patient_gender)

    print(f"\n==========================================")
    print(f"ESTIMASI HEMOGLOBIN (Hb): {hb_prediction:.2f} g/dL")
    print(f"==========================================\n")

    # ==========================================
    # 4. VISUALISASI HASIL PENGUJIAN
    # ==========================================
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Gambar Input")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_2d, cmap='gray') 
    plt.title("Prediksi Mask (UNBCSM)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(roi_image)
    plt.title(f"Hasil Potongan (ROI)\nPrediksi Hb: {hb_prediction:.2f} g/dL")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()