import os  # Thêm import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import get_model

# Thiết lập transforms cho tập test (giống với val)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def evaluate_model(data_dir, model_path, batch_size=32):
    # Thiết lập thiết bị
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tải dữ liệu test
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    class_names = test_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Number of test images: {len(test_dataset)}")
    
    # Tải mô hình
    model = get_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Dự đoán và lưu kết quả
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Tính confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Tính các chỉ số đánh giá
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Vẽ confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    evaluate_model('data/processed', 'models/best_model.pth')