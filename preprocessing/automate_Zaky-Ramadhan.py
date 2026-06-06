import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def run_preprocessing(input_path, output_dir):
    print("=== Memulai Proses Preprocessing Data (AqSolDB) ===")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Data mentah tidak ditemukan di: {input_path}")
        
    # 1. Data Loading
    df = pd.read_csv(input_path)
    
    # 2. Handling Missing Values
    df = df.dropna()
    
    # 3. Transformasi Target ke Klasifikasi Biner
    # Nilai Solubility (LogS) >= -3.0 dikategorikan sebagai larut/High Solubility (1), sisanya (0)
    if 'Solubility' in df.columns:
        df['target'] = (df['Solubility'] >= -3.0).astype(int)
    
    # 4. Drop Kolom Identitas Senyawa & Metadata yang tidak digunakan untuk pemodelan
    cols_to_drop = ['ID', 'Name', 'InChI', 'InChIKey', 'SMILES', 'Solubility', 'SD', 'Ocurrences', 'Group']
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)
    
    # 5. Feature Scaling
    X = df.drop(columns=['target'])
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    df_clean = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
    
    # 6. Menyimpan Hasil Preprocessing
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "data_clean.csv")
    df_clean.to_csv(output_file, index=False)
    
    print(f"=== Preprocessing Selesai! Data disimpan di: {output_file} ===")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Pastikan file CSV mentahan AqSolDB Anda di simpan dengan nama 'curated-solubility-dataset.csv' di folder namadataset_raw
    INPUT_DATA = os.path.join(BASE_DIR, "namadataset_raw", "curated-solubility-dataset.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessing", "namadataset_preprocessing")
    
    run_preprocessing(INPUT_DATA, OUTPUT_DIR)