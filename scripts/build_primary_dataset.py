import os
import sys
import cv2
import rawpy
import numpy as np
import pandas as pd
from scipy.stats import skew
import glob
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# KONFIGURASI PATH FOLDER & MODUL
# ==========================================
# Gunakan r"..." (raw string) agar Windows tidak salah membaca backslash (\)
BASE_GAMBAR = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\dataset\Hasil Pengambilan Data\Hasil Gambar"
BASE_GT = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\dataset\Hasil Pengambilan Data\Ground Truth"
OUTPUT_EXCEL = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\dataset\hasil_ekstraksi_fitur_primer2.xlsx"

# Konfigurasi Path untuk modul custom Anda
JALUR_UTAMA = r'C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA'
if JALUR_UTAMA not in sys.path:
    sys.path.append(JALUR_UTAMA)

# Import modul custom
try:
    from scripts.ColorCorrectionML import ColorCorrectionML
except ImportError:
    print(f"Error: Modul ColorCorrectionML tidak ditemukan di path {JALUR_UTAMA}")

# ==========================================
# FUNGSI-FUNGSI PIPELINE (Berdasarkan Notebook)
# ==========================================

def ambil_gambar_raw(dng_path):
    """
    1. Ambil Gambar: Membaca file DNG.
    """
    with rawpy.imread(dng_path) as raw:
        # Menyesuaikan dengan parameter pada code kalibrasi Anda: exp_shift=1.0
        rgb_image = raw.postprocess(use_camera_wb=True, exp_shift=1.0)
    return rgb_image

def get_mask_path(gt_folder, side):
    """ Mencari file mask biner (ekstensi bebas: .png/.jpg). """
    search_pattern = os.path.join(gt_folder, f"{side}.*")
    files = glob.glob(search_pattern)
    if files:
        return files[0]
    return None

def roi_cropping(image_rgb, mask):
    """
    2. ROI Cropping: Memotong gambar berdasarkan area Ground Truth.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_rgb, mask
    
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_rgb.shape[1], x + w + padding)
    y2 = min(image_rgb.shape[0], y + h + padding)
    
    cropped_img = image_rgb[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    return cropped_img, cropped_mask

def restoration(image_rgb, mask):
    """
    3. Restoration: Brightspot Denoising menggunakan Inpainting.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Pastikan inpaint hanya dilakukan di area mata
    bright_mask_in_eye = cv2.bitwise_and(bright_mask, mask)
    
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(img_bgr, bright_mask_in_eye, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    
    return inpainted_rgb

def color_correction(target_img, cc_img):
    """
    4. Color Correction: Koreksi warna berdasarkan referensi Color Checker
       menggunakan algoritma regresi PLS Polynomial.
    """
    # a. Konversi cc_img (RGB) ke BGR karena ColorCorrectionML membutuhkan input BGR
    bgr_cc = cv2.cvtColor(cc_img, cv2.COLOR_RGB2BGR)

    # b. Ekstraksi Color Chart dan Kalkulasi Bobot Model
    cc_ml = ColorCorrectionML(bgr_cc, chart='Classic', illuminant='D50')
    src_awal, _, _ = cc_ml.extract_color_chart()

    degree = 3
    model_pls, _ = cc_ml.compute_correction(
        show=False,
        method='pls',
        degree=degree,
        interactions_only=False,
        ncomp=10
    )

    poly = PolynomialFeatures(degree=degree, interaction_only=False)
    
    # c. Aplikasikan Model ke target_img (ROI Mata yang sudah bersih)
    img = target_img.copy() 
    h, w, c = img.shape
    img_flat = img.reshape(-1, 3)

    # Mengakali NotFittedError dengan memberikan dummy 1 piksel
    poly.fit(img_flat[:1])

    # Menggunakan wadah float32 untuk mencegah Integer Overflow
    img_corrected_flat = np.zeros_like(img_flat, dtype=np.float32)
    chunk_size = 1_000_000

    for i in range(0, len(img_flat), chunk_size):
        # Ubah potongan ke float32 sebelum masuk ke fungsi ML
        chunk = img_flat[i:i+chunk_size].astype(np.float32)

        chunk_poly = poly.transform(chunk)
        img_corrected_flat[i:i+chunk_size] = model_pls.predict(chunk_poly)

    # Kembalikan ke format gambar normal dengan batas 0-255
    corrected_img = np.clip(img_corrected_flat, 0, 255).astype(np.uint8).reshape(h, w, c)
    
    return corrected_img

def extract_features(image_rgb, mask):
    """
    5. Ekstraksi Fitur dari ruang warna RGB, HSV, LAB, serta Fitur Relatif.
    """
    features = {}
    valid_pixels = mask == 255
    
    # Inisialisasi channel
    channels = {}
    
    # RGB
    r, g, b = cv2.split(image_rgb)
    channels.update({'R': r, 'G': g, 'B': b})
    
    # HSV
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    channels.update({'H': h, 'S': s, 'V': v})
    
    # LAB
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b_lab = cv2.split(lab_image)
    channels.update({'L': l, 'A': a, 'B_lab': b_lab})
    
    # Ekstraksi statistik dasar
    for channel_name, channel_data in channels.items():
        pixel_values = channel_data[valid_pixels]
        
        if len(pixel_values) > 0:
            features[f'{channel_name}_Mean'] = np.mean(pixel_values)
            features[f'{channel_name}_Std'] = np.std(pixel_values)
            features[f'{channel_name}_Skew'] = skew(pixel_values) if len(pixel_values) > 2 else 0
        else:
            features[f'{channel_name}_Mean'] = 0
            features[f'{channel_name}_Std'] = 0
            features[f'{channel_name}_Skew'] = 0

    # --- EKSTRAKSI FITUR RELATIF (TAMBAHAN) ---
    
    # Ambil nilai Rata-rata RGB yang baru saja dihitung
    mean_R = features.get('R_Mean', 0)
    mean_G = features.get('G_Mean', 0)
    mean_B = features.get('B_Mean', 0)
    
    # 1. Erythema Index (Log Red - Log Green)
    # Ditambah 1.0 (epsilon) agar terhindar dari perhitungan log(0) jika piksel hitam sempurna
    features['Erythema_Index'] = np.log10(mean_R + 1.0) - np.log10(mean_G + 1.0)
    
    # 2. Red Ratio (R / (R + G + B))
    total_rgb = mean_R + mean_G + mean_B
    features['Red_Ratio'] = mean_R / total_rgb if total_rgb > 0 else 0
    
    # 3. High Hue Ratio (Rasio area yang sangat merah terhadap total area mata)
    h_pixels = channels['H'][valid_pixels]
    if len(h_pixels) > 0:
        # Dalam HSV OpenCV (0-179), warna merah dominan ada di sekitar 0-10 dan 170-179
        red_pixels_count = np.sum((h_pixels <= 10) | (h_pixels >= 170))
        features['High_Hue_Ratio'] = red_pixels_count / len(h_pixels)
    else:
        features['High_Hue_Ratio'] = 0
            
    return features

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================

def main():
    print("Memulai proses ekstraksi fitur sesuai pipeline Notebook...")
    all_features_data = []
    
    if not os.path.exists(BASE_GAMBAR):
        print(f"Error: Folder {BASE_GAMBAR} tidak ditemukan.")
        return

    subjek_list = sorted(os.listdir(BASE_GAMBAR))
    
    for subj in subjek_list:
        subj_path = os.path.join(BASE_GAMBAR, subj)
        if not os.path.isdir(subj_path):
            continue
            
        print(f"Memproses {subj}...")
        devices = ['Samsung A52s', 'iPhone 11']
        
        for device in devices:
            path_gambar = os.path.join(subj_path, device)
            path_gt = os.path.join(BASE_GT, subj, device)
            
            if not os.path.exists(path_gambar) or not os.path.exists(path_gt):
                continue
            
            sides = ['left', 'right']
            
            for side in sides:
                try:
                    # Path file
                    dng_file = os.path.join(path_gambar, f"{side}.dng")
                    cc_file = os.path.join(path_gambar, "cc.dng")
                    mask_file = get_mask_path(path_gt, side)
                    
                    if not os.path.exists(dng_file) or not os.path.exists(cc_file) or mask_file is None:
                        continue 
                    
                    # 1. Ambil Gambar
                    img_rgb = ambil_gambar_raw(dng_file)
                    cc_rgb = ambil_gambar_raw(cc_file)
                    
                    # Membaca mask biner murni.
                    # cv2.IMREAD_GRAYSCALE memaksa mask dibaca 1-channel (berjaga-jaga jika tersimpan sbg 3-channel).
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    
                    # Memastikan mutlak bahwa nilai background = 0 dan nilai foreground (mata) = 255.
                    # Ini mencegah error jika mask tersimpan dengan format piksel 0 dan 1.
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    
                    if mask.shape != img_rgb.shape[:2]:
                        mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # 2. ROI Cropping
                    roi_img, roi_mask = roi_cropping(img_rgb, mask)
                    
                    # 3. Restoration (Brightspot Denoising)
                    restored_img = restoration(roi_img, roi_mask)
                    
                    # 4. Color Correction
                    # Memasukkan CC_RGB sebagai referensi color checker, dan Restored_IMG untuk dikoreksi
                    corrected_img = color_correction(restored_img, cc_rgb)
                    
                    # 5. Ekstraksi Fitur (Dan bisa ditambah augmentasi/normalisasi jika ada)
                    features = extract_features(corrected_img, roi_mask)
                    
                    # Gabungkan Metadata
                    row_data = {
                        'Subjek': subj,
                        'Device': device,
                        'Posisi_Mata': side,
                    }
                    row_data.update(features)
                    
                    all_features_data.append(row_data)
                    print(f"  -> Sukses: {device} - {side}")
                    
                except Exception as e:
                    print(f"  [Error] Gagal memproses {subj} - {device} - {side}: {e}")

    # Simpan Data
    if all_features_data:
        df = pd.DataFrame(all_features_data)
        df.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\nSelesai! Berhasil disimpan ke: {OUTPUT_EXCEL}")
        print(f"Total data diproses: {len(df)} gambar")
    else:
        print("\nTidak ada data yang diekstrak.")

if __name__ == "__main__":
    main()