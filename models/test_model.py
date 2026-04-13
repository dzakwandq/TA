from load_model_TA import loadmodel2, read_image


# path model hasil training (ganti sesuai punya kamu)
model_path = r'C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\models\weights\simulasi\model_simulasi_terbaik.keras'

# load model (bangun arsitektur + muat bobot)
model = loadmodel2(model_path)

# path ke gambar yang mau diuji
image_path = r'C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\dataset\Dataset Eyedefy\India\2\20200124_154320.jpg'

# prediksi segmentasi
image, gray_img, mask, blended = read_image(model, image_path)

# tampilkan hasil (opsional)
import cv2
cv2.imshow("Original", image)
cv2.imshow("Segmented Mask", mask)
cv2.imshow("Result (ROI)", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()