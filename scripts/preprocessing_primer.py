import cv2
import numpy as np

def deteksi_mata(img_rgb):
    """
    Fungsi untuk mendeteksi konjungtiva dan mengembalikan koordinat Bounding Box.
    Menggunakan thresholding warna merah pada format HSV.
    """
    # 1. Konversi RGB ke BGR (karena OpenCV menggunakan format BGR)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 2. Thresholding warna merah (di ujung angka 0-15 dan 160-180)
    lower_red1 = np.array([0, 40, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 40, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_kemerahan = mask1 + mask2

    # 3. Operasi Morfologi untuk membersihkan noise
    kernel = np.ones((7, 7), np.uint8)
    mask_bersih = cv2.morphologyEx(mask_kemerahan, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_bersih = cv2.morphologyEx(mask_bersih, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Deteksi Kontur
    contours, _ = cv2.findContours(mask_bersih, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        kontur_terbesar = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(kontur_terbesar)

        # print(f"Area konjungtiva yang terdeteksi: {area} piksel")

        if area > 1000:
            x, y, w, h = cv2.boundingRect(kontur_terbesar)

            # Tambahkan margin sesuai referensi
            margin_atas = int(h * 2)
            margin_bawah = int(h * 0.2)
            margin_samping = int(w * 0.5)

            img_h, img_w, _ = img_rgb.shape

            x_min = max(0, x - margin_samping)
            y_min = max(0, y - margin_atas)
            x_max = min(img_w, x + w + margin_samping)
            y_max = min(img_h, y + h + margin_bawah)

            return (x_min, y_min, x_max, y_max)
    
    # Return None jika konjungtiva gagal dideteksi (misal area < 1000)
    return None

def restoration(roi_img, roi_mask=None):
    """
    Fungsi Brightspot Denoising (Restoration) menggunakan algoritma Inpainting.
    Mencari area sangat terang (pantulan lampu/cahaya) lalu menambalnya (inpaint).
    """
    # Ubah ke Grayscale untuk mencari intensitas cahaya paling tinggi
    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    
    # Deteksi area pantulan cahaya (Brightness > 220)
    _, bright_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    # Batasi deteksi brightspot hanya pada area mask konjungtiva (jika diberikan)
    if roi_mask is not None:
        bright_mask = cv2.bitwise_and(bright_mask, roi_mask)
        
    # Perlebar (dilate) sedikit mask pantulan agar area transisinya ikut diperbaiki
    kernel = np.ones((3,3), np.uint8)
    bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
    
    # Lakukan Inpainting menggunakan algoritma Telea
    restored_img = cv2.inpaint(roi_img, bright_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return restored_img

def color_correction(target_img, cc_img):
    """
    Fungsi Color Correction menggunakan metode PLS (Partial Least Squares) dan Polynomial Features.
    Mengekstrak color chart dari gambar cc_img dan menerapkan koreksi ke target_img.
    Memanfaatkan chunking untuk mencegah Out of Memory (OOM) saat memproses citra beresolusi tinggi.
    """
    from scripts.ColorCorrectionML import ColorCorrectionML
    from sklearn.preprocessing import PolynomialFeatures

    bgr_cc = cv2.cvtColor(cc_img, cv2.COLOR_RGB2BGR)
    cc_ml = ColorCorrectionML(bgr_cc, chart='Classic', illuminant='D50')
    src_awal, _, _ = cc_ml.extract_color_chart()

    degree = 3
    model_pls, _ = cc_ml.compute_correction(show=False, method='pls', degree=degree, interactions_only=False, ncomp=10)
    poly = PolynomialFeatures(degree=degree, interaction_only=False)
    
    img = target_img.copy() 
    h, w, c = img.shape
    img_flat = img.reshape(-1, 3)

    poly.fit(img_flat[:1])
    img_corrected_flat = np.zeros_like(img_flat, dtype=np.float32)
    chunk_size = 1_000_000

    for i in range(0, len(img_flat), chunk_size):
        chunk = img_flat[i:i+chunk_size].astype(np.float32)
        chunk_poly = poly.transform(chunk)
        img_corrected_flat[i:i+chunk_size] = model_pls.predict(chunk_poly)

    corrected_img = np.clip(img_corrected_flat, 0, 255).astype(np.uint8).reshape(h, w, c)
    return corrected_img