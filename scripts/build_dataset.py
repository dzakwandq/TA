import os
import glob
import cv2
import pandas as pd
import numpy as np

from feature_extraction import extract_color_features

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
    return str(val).strip().capitalize() # Contoh output: 'Male', 'Female', 'M', 'F'

def classify_anemia(hb, age, gender):
    """
    Mengklasifikasikan tingkat Anemia berdasarkan standar WHO.
    Satuan yang digunakan di sini adalah g/dL (sesuai data excel).
    
    Return:
    3 = Normal, 2 = Mild, 1 = Moderate, 0 = Severe
    """
    if pd.isna(hb) or pd.isna(age) or gender == "Unknown":
        return np.nan
    
    # Standarisasi pembacaan string gender
    gender_lower = str(gender).strip().lower()
    is_male = gender_lower in ['m', 'male', 'laki-laki']
    is_female = gender_lower in ['f', 'female', 'perempuan', 'w']
    
    # 1. Children 6 - 59 months (Kita asumsikan < 5 tahun)
    if age < 5:
        if hb >= 11.0: return 3
        elif hb >= 10.0: return 2
        elif hb >= 7.0: return 1
        else: return 0
        
    # 2. Children 5 - 11 years
    elif 5 <= age <= 11:
        if hb >= 11.5: return 3
        elif hb >= 11.0: return 2
        elif hb >= 8.0: return 1
        else: return 0
        
    # 3. Children 12 - 14 years
    elif 12 <= age <= 14:
        if hb >= 12.0: return 3
        elif hb >= 11.0: return 2
        elif hb >= 8.0: return 1
        else: return 0
        
    # 4. Adult Men (15+ years)
    elif age >= 15 and is_male:
        if hb >= 13.0: return 3
        elif hb >= 11.0: return 2
        elif hb >= 8.0: return 1
        else: return 0
        
    # 5. Non-pregnant Adult Women (15+ years)
    elif age >= 15 and is_female:
        if hb >= 12.0: return 3
        elif hb >= 11.0: return 2
        elif hb >= 8.0: return 1
        else: return 0
        
    return np.nan

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
            
            # PERBAIKAN 1: Memastikan ID yang float (misal 1.0) diubah jadi integer lalu string '1'
            try:
                subj_id = str(int(float(raw_number))).strip()
            except ValueError:
                subj_id = str(raw_number).strip()
            
            # Ambil nilai-nilainya
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
    OUTPUT_EXCEL = r"C:\Folder Dzakwan\Folder Dzakwan\Keperluan TA\dataset\hasil_ekstraksi_fitur_tanpaClass.xlsx"
    
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
            
            # PERBAIKAN 2: Mencegah KeyError jika folder ada tapi datanya tidak ada di Excel
            if subj_id not in patient_info[country]:
                print(f"Melewati Subjek {subj_id} (Data tidak ditemukan di Excel)")
                continue
                
            subj_data = patient_info[country][subj_id]
            hgb_value = subj_data['Hgb']
            age_value = subj_data['Age']
            gender_value = subj_data['Gender']
            
            anemia_class = classify_anemia(hgb_value, age_value, gender_value)

            img_path, mask_path = find_image_and_mask(subj_folder)
            
            if img_path and mask_path:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if len(mask_img.shape) == 3 and mask_img.shape[2] == 4:
                    _, mask_2d = cv2.threshold(mask_img[:, :, 3], 0, 255, cv2.THRESH_BINARY)
                else:
                    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY) if len(mask_img.shape) == 3 else mask_img
                    _, mask_2d = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                
                mask_2d = cv2.resize(mask_2d, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                features = extract_color_features(img_rgb, mask_2d)

                row_data = {
                    'Country': country,
                    'Subject_ID': int(subj_id),
                    'Age': age_value,          
                    'Gender': gender_value,     
                    'mean_R': features['mean_R'],
                    'mean_G': features['mean_G'],
                    'mean_B': features['mean_B'],
                    'mean_L': features['mean_L'],
                    'mean_a': features['mean_a'],
                    'mean_b': features['mean_b'],
                    'mean_H': features['mean_H'],
                    'mean_S': features['mean_S'],
                    'mean_V': features['mean_V'],
                    'Hemoglobin': hgb_value,
                    #'Anemia_Class': anemia_class
                }
                final_data.append(row_data)
            else:
                print(f"Melewati Subjek {subj_id} (Gambar/Mask tidak lengkap)")

    df_final = pd.DataFrame(final_data)
    df_final.to_excel(OUTPUT_EXCEL, index=False)
    
    print(f"\nSelesai! File disimpan di: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()