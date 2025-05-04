import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2):
    model = models.resnet34(pretrained=True)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Thay đổi lớp cuối cùng
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
