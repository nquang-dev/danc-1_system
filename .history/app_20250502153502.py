import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model import get_model
from src.visualization import apply_cam
import tempfile
import os
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as reportlab_colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use('Agg')
import uuid
from database import User, Patient, Analysis, get_db

# Thiết lập trang
st.set_page_config(
    page_title="Phát hiện lao phổi qua X-quang", 
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🫁"
)

# Kiểm tra đăng nhập
if "user_info" not in st.session_state:
    st.warning("Bạn cần đăng nhập để sử dụng ứng dụng")
    st.button("Đi đến trang đăng nhập", on_click=lambda: st.switch_page("auth.py"))
    st.stop()

# Đăng ký font hỗ trợ tiếng Việt
font_paths = [
    os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf'),
    os.path.join(os.path.dirname(__file__), 'fonts', 'DejaVuSans.ttf'),
    'DejaVuSans.ttf'
]

font_registered = False
for font_path in font_paths:
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
            font_registered = True
            break
        except Exception as e:
            st.warning(f"Không thể đăng ký font từ {font_path}: {e}")

if not font_registered:
    st.warning("Không thể đăng ký font DejaVuSans. Sẽ sử dụng font mặc định.")

# Import CSS từ file riêng
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Tải mô hình
@st.cache_resource
def load_model():
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Tiền xử lý ảnh
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tạo báo cáo PDF
def create_pdf_report(image, cam_image, prediction, prob_normal, prob_tb, process_time, filename=None, patient_info=None):
    buffer = io.BytesIO()
    
    # Tạo document PDF
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Tạo style cho văn bản
    styles = getSampleStyleSheet()
    
    # Kiểm tra xem font DejaVuSans đã được đăng ký chưa
    vietnamese_font = 'DejaVuSans'
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        styles.add(ParagraphStyle(name='Vietnamese', fontName=vietnamese_font, fontSize=12))
    else:
        # Nếu font không có, sử dụng font mặc định
        styles.add(ParagraphStyle(name='Vietnamese', fontName='Helvetica', fontSize=12))
    
    # Danh sách các phần tử trong PDF
    elements = []
    
    # Tiêu đề
    title_style = styles["Heading1"]
    title_style.alignment = 1  # Center
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        title_style.fontName = vietnamese_font
    elements.append(Paragraph("KẾT QUẢ PHÂN TÍCH X-QUANG PHỔI", title_style))
    elements.append(Spacer(1, 20))
    
    # Ngày giờ phân tích
    date_style = styles["Normal"]
    date_style.alignment = 1  # Center
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        date_style.fontName = vietnamese_font
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    elements.append(Paragraph(f"Ngày giờ phân tích: {current_time}", date_style))
    elements.append(Spacer(1, 20))
    
    # Thông tin bệnh nhân nếu có
    if patient_info:
        patient_style = styles["Heading2"]
        if vietnamese_font in pdfmetrics.getRegisteredFontNames():
            patient_style.fontName = vietnamese_font
        elements.append(Paragraph("THÔNG TIN BỆNH NHÂN", patient_style))
        elements.append(Spacer(1, 10))
        
        patient_data = [
            ["Thông tin", "Chi tiết"],
            ["Mã bệnh nhân", patient_info["patient_code"]],
            ["Họ và tên", patient_info["full_name"]],
            ["Tuổi", str(patient_info["age"])],
            ["Giới tính", patient_info["gender"]]
        ]
        
        if patient_info.get("address"):
            patient_data.append(["Địa chỉ", patient_info["address"]])
        
        if patient_info.get("phone"):
            patient_data.append(["Số điện thoại", patient_info["phone"]])
        
        patient_table = Table(patient_data, colWidths=[200, 200])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), reportlab_colors.grey),
            ('TEXTCOLOR', (0, 0), (1, 0), reportlab_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (1, 0), vietnamese_font if vietnamese_font in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
            ('FONTNAME', (0, 1), (0, -1), vietnamese_font if vietnamese_font in pdfmetrics.getRegisteredFontNames() else 'Helvetica'),
            ('FONTNAME', (1, 1), (1, -1), vietnamese_font if vietnamese_font in pdfmetrics.getRegisteredFontNames() else 'Helvetica'),
        ]))
        
        elements.append(patient_table)
        elements.append(Spacer(1, 20))
    
    if filename:
        file_style = styles["Normal"]
        if vietnamese_font in pdfmetrics.getRegisteredFontNames():
            file_style.fontName = vietnamese_font
        elements.append(Paragraph(f"Tên file: {filename}", file_style))
        elements.append(Spacer(1, 10))
    
    # Lưu ảnh gốc và ảnh CAM
    img_path = tempfile.mktemp(suffix='.png')
    cam_path = tempfile.mktemp(suffix='.png')
    
    # Chuyển đổi ảnh PIL sang định dạng phù hợp
    image.save(img_path)
    
    # Nếu cam_image là mảng numpy, chuyển thành PIL Image
    if isinstance(cam_image, np.ndarray):
        cam_image_pil = Image.fromarray(cam_image)
        cam_image_pil.save(cam_path)
    else:
        cam_image.save(cam_path)
    
    # Thêm ảnh vào PDF
    heading_style = styles["Heading2"]
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        heading_style.fontName = vietnamese_font
    
    elements.append(Paragraph("Ảnh X-quang gốc:", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(RLImage(img_path, width=400, height=300))
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Ảnh phân tích (CAM):", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(RLImage(cam_path
