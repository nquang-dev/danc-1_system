import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import time

class CAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Đăng ký hook
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Zero grads
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights
        gradients = self.gradients.squeeze()
        activations = self.activations.squeeze()
        
        # Take the mean of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        # Weighted sum of activation maps
        cam = torch.sum(weights * activations, dim=0)
        
        # ReLU
        cam = torch.maximum(cam, torch.tensor(0.))
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam.cpu().numpy()

def apply_cam(image_path, model, preprocess, last_conv_layer):
    start_time = time.time()
    
    # Tải và tiền xử lý ảnh
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)
    
    # Thiết lập thiết bị
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    
    # Dự đoán
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        prob_normal = probabilities[0].item()
        prob_tb = probabilities[1].item()
    
    # Chỉ tạo CAM nếu dự đoán là TB (lao phổi)
    if pred.item() == 1:
        # Tạo CAM
        cam_generator = CAM(model, last_conv_layer)
        cam = cam_generator.generate_cam(input_tensor)
        
        # Chuyển đổi ảnh gốc
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (224, 224))
        
        # Tạo heatmap
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Kết hợp ảnh gốc và heatmap
        alpha = 0.4
        superimposed = heatmap * alpha + img_np * (1 - alpha)
        superimposed = np.uint8(superimposed)
        
        # Chuyển thành PIL Image
        superimposed = Image.fromarray(superimposed)
    else:
        superimposed = img.resize((224, 224))
    
    process_time = time.time() - start_time
    
    return superimposed, pred.item(), prob_normal, prob_tb, process_time