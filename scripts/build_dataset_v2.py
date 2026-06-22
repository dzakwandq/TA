import os
import glob
import cv2
import pandas as pd
import numpy as np
from scipy.stats import skew

def clean_hgb_value(val):
    """ Membersihkan nilai Hgb dari Excel. """
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace('"', '').replace(',', '.').strip()
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def clean_gender_value(val):
    """ Membersihkan string gender (misal: menghapus spasi berlebih) """
    if pd.isna(val):
        return "Unknown" # Jika kosong
    return str(val).strip().capitalize()

def classify_anemia(hb, age, gender):
    """
    Mengklasifikasikan tingkat Anemia berdasarkan standar WHO.
    Satuan yang digunakan di sini adalah g/dL (sesuai data excel).
    
    Return (Binary Class):
    1 = Anemia (Termasuk Mild, Moderate, Severe)
    0 = Non-Anemia (Normal)
    """
    if pd.isna(hb) or pd.isna(age) or gender == "Unknown":
        return np.nan
    
    # Standarisasi pembacaan string gender
    gender_lower = str(gender).strip().lower()
    is_male = gender_lower in ['m', 'male', 'laki-laki']
    is_female = gender_lower in ['f', 'female', 'perempuan', 'w']
    
    # 1. Children 6 - 59 months (< 5 tahun)
    if age < 5:
        return 0 if hb >= 11.0 else 1
        
    # 2. Children 5 - 11 years
    elif 5 <= age <= 11:
        return 0 if hb >= 11.5 else 1
        
    # 3. Children 12 - 14 years
    elif 12 <= age <= 14:
        return 0 if hb >= 12.0 else 1
        
    # 4. Adult Men (15+ years)
    elif age >= 15 and is_male:
        return 0 if hb >= 13.0 else 1
        
    # 5. Non-pregnant Adult Women (15+ years)
    elif age >= 15 and is_female:
        return 0 if hb >= 12.0 else 1
        
    return np.nan

def extract_features(image_rgb, mask):
    """
    Mengekstrak fitur statistik (Mean, Std, Skewness) dari RGB, HSV, dan LAB.
    Hanya piksel di dalam mask (bernilai 255) yang dihitung.
    Telah ditambahkan fitur relatif selaras dengan pipeline primer (tanpa pembulatan).
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
            features[f'{channel_name}_Skew'] = skew(pixel_values) if len(pixel_values) > 2 else 0.0
        else:
            features[f'{channel_name}_Mean'] = 0.0
            features[f'{channel_name}_Std'] = 0.0
            features[f'{channel_name}_Skew'] = 0.0

    # --- EKSTRAKSI FITUR RELATIF (TAMBAHAN) ---
    
    # Ambil nilai Rata-rata RGB yang baru saja dihitung
    mean_R = features.get('R_Mean', 0)
    mean_G = features.get('G_Mean', 0)
    mean_B = features.get('B_Mean', 0)
    
    # 1. Erythema Index (Log Red - Log Green)
    features['Erythema_Index'] = np.log10(mean_R + 1.0) - np.log10(mean_G + 1.0)
    
    # 2. Red Ratio (R / (R + G + B))
    total_rgb = mean_R + mean_G + mean_B
    features['Red_Ratio'] = (mean_R / total_rgb) if total_rgb > 0 else 0.0
    
    # 3. High Hue Ratio (Rasio area yang sangat merah terhadap total area mata)
    h_pixels = channels['H'][valid_pixels]
    if len(h_pixels) > 0:
        # Dalam HSV OpenCV (0-179), warna merah dominan ada di sekitar 0-10 dan 170-179
        red_pixels_count = np.sum((h_pixels <= 10) | (h_pixels >= 170))
        features['High_Hue_Ratio'] = red_pixels_count / len(h_pixels)
    else:
        features['High_Hue_Ratio'] = 0.0
            
    return features

def load_patient_data(excel_india_path, excel_italy_path):
    """
    baca data India dan Italy (Hgb, Umur, Gender).
    format output: { 'India': { '1': {'Hgb': 12.2, 'Age': 25, 'Gender': 'M'}} }
    """
    patient_data = {'India': {}, 'Italy': {}}
    
    def process_df(df, country_name):
        col_age = 'Age' if 'Age' in df.columns else 'Umur' 
        col_gender = 'Gender' if 'Gender' in df.columns else 'Sex'
        
        for _, row in df.iterrows():
            raw_number = row['Number']
            if pd.isna(raw_number):
                continue
            
            try:
                subj_id = str(int(float(raw_number))).strip()
            except ValueError:
                subj_id = str(raw_number).strip()
            
            hgb = row['Hgb'] if 'Hgb' in df.columns else np.nan
            age = row[col_age] if col_age in df.columns else np.nan
            gender = row[col_gender] if col_gender in df.columns else "Unknown"
            
            patient_data[country_name][subj_id] = {
                'Hgb': clean_hgb_value(hgb),
                'Age': age,
                'Gender': clean_gender_value(gender)
            }

    if os.path.exists(excel_india_path):
        df_india = pd.read_excel(excel_india_path)
        process_df(df_india, 'India')
            
    if os.path.exists(excel_italy_path):
        df_italy = pd.read_excel(excel_italy_path)
        process_df(df_italy, 'Italy')
            
    return patient_data

def find_image_and_mask(subject_folder):
    mask_path = None
    image_path = None
    
    search_mask = glob.glob(os.path.join(subject_folder, '*forniceal_palpebral*.*'))
    if search_mask:
        mask_path = search_mask[0]
        
    all_jpgs = glob.glob(os.path.join(subject_folder, '*.jpg'))
    for jpg in all_jpgs:
        if 'forniceal' not in jpg and 'palpebral' not in jpg:
            image_path = jpg
            break
            
    return image_path, mask_path

def main():
    DATASET_DIR = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\dataset\Dataset Eyedefy"
    EXCEL_INDIA = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\dataset\Dataset Eyedefy\India\India.xlsx"
    EXCEL_ITALY = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\dataset\Dataset Eyedefy\Italy\Italy.xlsx"
    OUTPUT_EXCEL = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\dataset\hasil_ekstraksi_fitur_Eyedefy2.xlsx"
    
    print("Membaca data pasien...")
    
    patient_info = load_patient_data(EXCEL_INDIA, EXCEL_ITALY)
    
    final_data = []
    countries = ['India', 'Italy']
    
    for country in countries:
        country_dir = os.path.join(DATASET_DIR, country)
        if not os.path.exists(country_dir):
            print(f"Folder negara tidak ditemukan: {country_dir}")
            continue
            
        print(f"\nMemproses data dari: {country}")
        subject_folders = [f.path for f in os.scandir(country_dir) if f.is_dir()]
        
        for subj_folder in subject_folders:
            subj_id = os.path.basename(subj_folder)
            
            if subj_id not in patient_info[country]:
                print(f"Melewati Subjek {subj_id} (Data tidak ditemukan di Excel)")
                continue
                
            subj_data = patient_info[country][subj_id]
            hgb_value = subj_data['Hgb']
            age_value = subj_data['Age']
            gender_value = subj_data['Gender']
            
            # Klasifikasi Anemia (1 = Anemia, 0 = Normal)
            anemia_class = classify_anemia(hgb_value, age_value, gender_value)

            img_path, mask_path = find_image_and_mask(subj_folder)
            
            if img_path and mask_path:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

                if mask_img.dtype == np.uint16:
                    mask_img = (mask_img / 256).astype(np.uint8)

                if len(mask_img.shape) == 3 and mask_img.shape[2] == 4:
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2RGBA)
                else:
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
                
                mask_hsv = cv2.cvtColor(mask_img, cv2.COLOR_RGB2HSV)
    
                saturation = mask_hsv[:, :, 1]
                
                if np.min(mask_img) == 0:
                    _, mask_2d = cv2.threshold(mask_img[:, :, 3], 0, 255, cv2.THRESH_BINARY)
                else:    
                    _, mask_2d = cv2.threshold(saturation, 1, 255, cv2.THRESH_BINARY)

                kernel = np.ones((6,6), np.uint8)
                mask_2d = cv2.morphologyEx(mask_2d, cv2.MORPH_CLOSE, kernel)
                
                # Resize mask
                mask_2d = cv2.resize(mask_2d, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # ---> EKSTRAKSI 30 FITUR WARNA (Termasuk Fitur Relatif) <---
                features = extract_features(img_rgb, mask_2d)

                # Gabungkan Metadata dan Fitur Warna
                row_data = {
                    'Country': country,
                    'Subject_ID': int(subj_id),
                    'Age': age_value,          
                    'Gender': gender_value,     
                    'Hemoglobin': hgb_value,
                    'Anemia_Class': anemia_class # Menambahkan kembali Class
                }
                row_data.update(features) # Memasukkan otomatis 30 kolom warna
                
                final_data.append(row_data)
            else:
                print(f"Melewati Subjek {subj_id} (Gambar/Mask tidak lengkap)")

    # Simpan ke Excel
    if final_data:
        df_final = pd.DataFrame(final_data)
        df_final.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\nSelesai! File disimpan di: {OUTPUT_EXCEL}")
    else:
        print("\nTidak ada data yang diproses.")

if __name__ == "__main__":
    main()