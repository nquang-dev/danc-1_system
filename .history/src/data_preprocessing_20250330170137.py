import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_data(raw_dir, processed_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Tạo thư mục
    for split in ['train', 'val', 'test']:
        for cls in ['Normal', 'Tuberculosis']:
            os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)
    
    # Phân chia dữ liệu
    for cls in ['Normal', 'Tuberculosis']:
        files = os.listdir(os.path.join(raw_dir, cls))
        train_files, temp_files = train_test_split(files, train_size=train_ratio, random_state=42)
        val_size = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(temp_files, train_size=val_size, random_state=42)
        
        # Copy files
        for file in train_files:
            shutil.copy(os.path.join(raw_dir, cls, file), 
                       os.path.join(processed_dir, 'train', cls, file))
        for file in val_files:
            shutil.copy(os.path.join(raw_dir, cls, file), 
                       os.path.join(processed_dir, 'val', cls, file))
        for file in test_files:
            shutil.copy(os.path.join(raw_dir, cls, file), 
                       os.path.join(processed_dir, 'test', cls, file))

if __name__ == "__main__":
    split_data('data/raw', 'data/processed')
