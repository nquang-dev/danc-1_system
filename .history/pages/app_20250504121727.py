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
    elements.append(RLImage(cam_path, width=400, height=300))
    elements.append(Spacer(1, 20))
    
    # Kết quả phân tích
    elements.append(Paragraph("KẾT QUẢ CHẨN ĐOÁN:", heading_style))
    elements.append(Spacer(1, 10))
    
    if prediction == 1:
        result_text = "PHÁT HIỆN DẤU HIỆU LAO PHỔI"
        result_color = reportlab_colors.red
    else:
        result_text = "KHÔNG PHÁT HIỆN DẤU HIỆU LAO PHỔI"
        result_color = reportlab_colors.green
    
    result_style = ParagraphStyle(
        name='ResultStyle',
        parent=styles["Heading2"],
        textColor=result_color,
        alignment=1  # Center
    )
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        result_style.fontName = vietnamese_font
    elements.append(Paragraph(result_text, result_style))
    elements.append(Spacer(1, 20))
    
    # Thông tin chi tiết
    data = [
        ["Thông số", "Giá trị"],
        ["Xác suất bình thường", f"{prob_normal:.2%}"],
        ["Xác suất lao phổi", f"{prob_tb:.2%}"],
        ["Thời gian xử lý", f"{process_time:.2f} giây"]
    ]

    table = Table(data, colWidths=[200, 200])

    # Xác định font sẽ sử dụng
    use_font = vietnamese_font if vietnamese_font in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
    bold_font = vietnamese_font if vietnamese_font in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold'

    table.setStyle(TableStyle([
        # Định dạng hàng tiêu đề
        ('BACKGROUND', (0, 0), (1, 0), reportlab_colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), reportlab_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), bold_font),  # Tiêu đề sử dụng font in đậm
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        
        # Định dạng nội dung
        ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
        
        # Áp dụng font cho tất cả các ô nội dung
        ('FONTNAME', (0, 1), (0, -1), use_font),  # Cột 1 (Thông số)
        ('FONTNAME', (1, 1), (1, -1), use_font),  # Cột 2 (Giá trị)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Thông tin bác sĩ
    if "user_info" in st.session_state:
        doctor_info = f"Bác sĩ phân tích: {st.session_state.user_info['full_name']}"
        elements.append(Paragraph(doctor_info, styles["Vietnamese"]))
        elements.append(Spacer(1, 10))
    
    # Lưu ý
    note_heading_style = styles["Heading3"]
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        note_heading_style.fontName = vietnamese_font
    elements.append(Paragraph("Lưu ý:", note_heading_style))
    elements.append(Paragraph("Kết quả này chỉ mang tính chất tham khảo. Vui lòng tham khảo ý kiến của bác sĩ chuyên khoa để có chẩn đoán chính xác.", styles["Vietnamese"]))
    
    # Xây dựng PDF
    doc.build(elements)
    
    # Xóa file tạm
    os.unlink(img_path)
    os.unlink(cam_path)
    
    buffer.seek(0)
    return buffer

# Lưu kết quả phân tích vào database
def save_analysis_to_db(image, cam_image, prediction, prob_normal, prob_tb, process_time, filename, patient_id=None, notes=None):
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    os.makedirs("static/uploads", exist_ok=True)
    
    # Tạo tên file duy nhất
    unique_id = str(uuid.uuid4())
    image_filename = f"{unique_id}_original.png"
    cam_filename = f"{unique_id}_cam.png"
    
    # Đường dẫn lưu trữ
    image_path = os.path.join("static/uploads", image_filename)
    cam_path = os.path.join("static/uploads", cam_filename)
    
    # Lưu ảnh
    image.save(image_path)
    
    # Nếu cam_image là mảng numpy, chuyển thành PIL Image
    if isinstance(cam_image, np.ndarray):
        cam_image_pil = Image.fromarray(cam_image)
        cam_image_pil.save(cam_path)
    else:
        cam_image.save(cam_path)
    
    # Lưu vào database
    db = next(get_db())
    new_analysis = Analysis(
        image_path=image_path,
        cam_image_path=cam_path,
        prediction=prediction,
        probability_normal=prob_normal,
        probability_tb=prob_tb,
        process_time=process_time,
        notes=notes,
        patient_id=patient_id,
        doctor_id=st.session_state.user_info["id"]
    )
    
    db.add(new_analysis)
    db.commit()
    
    return new_analysis.id

# Header với logo và tiêu đề
st.markdown("""
<div class="header">
    <div class="logo-container">
        <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" class="logo">
    </div>
    <div class="title-container">
        <h1>PHÁT HIỆN LAO PHỔI QUA ẢNH X-QUANG</h1>
        <p class="subtitle">Ứng dụng trí tuệ nhân tạo trong chẩn đoán hình ảnh y tế</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar với thông tin người dùng
with st.sidebar:
    st.markdown(f"**Xin chào, {st.session_state.user_info['full_name']}!**")
    st.markdown("---")
    
    if st.button("Quản lý bệnh nhân"):
        st.switch_page("pages/patient_management.py")
    
    if st.session_state.user_info["is_admin"] and st.button("Trang quản trị"):
        st.switch_page("pages/admin.py")
    
    st.markdown("---")
    if st.button("Đăng xuất"):
        del st.session_state.user_info
        st.switch_page("auth.py")

# Tải mô hình
try:
    with st.spinner("Đang tải mô hình AI..."):
        model = load_model()
        last_conv_layer = model.layer4[-1]
    st.success("✅ Mô hình đã được tải thành công!")
except Exception as e:
    st.error(f"❌ Không thể tải mô hình: {e}")
    st.stop()

# Tạo tabs với thiết kế mới
tabs = st.tabs([
    "🔍 Phân tích một ảnh", 
    "📊 Phân tích nhiều ảnh",
    "ℹ️ Thông tin"
])

with tabs[0]:
    st.markdown("""
    <div class="tab-description">
        Tải lên một ảnh X-quang phổi để phân tích và phát hiện dấu hiệu lao phổi.
    </div>
    """, unsafe_allow_html=True)
    
    # Chọn bệnh nhân (nếu có)
    db = next(get_db())
    
    if st.session_state.user_info["is_admin"]:
        # Admin có thể xem tất cả bệnh nhân
        patients = db.query(Patient).all()
    else:
        # Bác sĩ chỉ xem bệnh nhân của mình
        patients = db.query(Patient).filter(Patient.doctor_id == st.session_state.user_info["id"]).all()
    
    patient_options = [("none", "Không chọn bệnh nhân")] + [(str(p.id), f"{p.patient_code} - {p.full_name}") for p in patients]
    
    # Nếu có bệnh nhân được chọn từ trang quản lý bệnh nhân
    selected_patient_id = None
    if "selected_patient_id" in st.session_state:
        selected_patient_id = st.session_state.selected_patient_id
        # Xóa khỏi session state để không ảnh hưởng lần sau
        del st.session_state.selected_patient_id
    
    selected_patient = st.selectbox(
        "Chọn bệnh nhân (không bắt buộc):",
        options=[id for id, _ in patient_options],
        format_func=lambda x: next((name for id, name in patient_options if id == x), ""),
        index=next((i for i, (id, _) in enumerate(patient_options) if id == str(selected_patient_id)), 0) if selected_patient_id else 0
    )
    
    patient_info = None
    if selected_patient != "none":
        patient = db.query(Patient).filter(Patient.id == int(selected_patient)).first()
        if patient:
            patient_info = {
                "id": patient.id,
                "patient_code": patient.patient_code,
                "full_name": patient.full_name,
                "age": patient.age,
                "gender": patient.gender,
                "address": patient.address,
                "phone": patient.phone
            }
            
            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-header">Thông tin bệnh nhân</div>
                <div class="info-card-content">
                    <div class="info-item">
                        <div class="info-label">Mã bệnh nhân:</div>
                        <div class="info-value">{patient.patient_code}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Họ và tên:</div>
                        <div class="info-value">{patient.full_name}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Tuổi:</div>
                        <div class="info-value">{patient.age}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Giới tính:</div>
                        <div class="info-value">{patient.gender}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">Tải lên ảnh X-quang phổi</div>', unsafe_allow_html=True)
        
        upload_placeholder = st.empty()
        with upload_placeholder.container():
            uploaded_file = st.file_uploader(
                "Kéo thả hoặc nhấp để chọn ảnh X-quang", 
                type=["jpg", "jpeg", "png"], 
                key="single_upload",
                help="Hỗ trợ các định dạng: JPG, JPEG, PNG"
            )
        
        if uploaded_file:
            # Hiển thị ảnh gốc trong khung đẹp hơn
            image = Image.open(uploaded_file).convert('RGB')
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="image-caption">Ảnh: {uploaded_file.name}</div>', unsafe_allow_html=True)
            
            # Ghi chú (không bắt buộc)
            notes = st.text_area("Ghi chú (không bắt buộc):", height=100)
            
            col1_buttons = st.columns(2)
            with col1_buttons[0]:
                analyze_button = st.button("🔍 Phân tích ảnh", key="analyze_single", use_container_width=True)
            with col1_buttons[1]:
                reset_button = st.button("🔄 Làm mới", key="reset_single", use_container_width=True)
            
            if reset_button:
                st.session_state.needs_rerun = True
            
            if analyze_button:
                with st.spinner("⏳ Đang phân tích ảnh..."):
                    # Lưu ảnh tạm thời
                    temp_dir = tempfile.mkdtemp()
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    image.save(temp_file_path)
                    
                    # Phân tích ảnh
                    cam_image, prediction, prob_normal, prob_tb, process_time = apply_cam(
                        temp_file_path, model, preprocess, last_conv_layer)
                    
                    # Lưu kết quả vào database
                    patient_id = int(selected_patient) if selected_patient != "none" else None
                    analysis_id = save_analysis_to_db(
                        image, cam_image, prediction, prob_normal, prob_tb, process_time, 
                        uploaded_file.name, patient_id, notes
                    )
                    
                    with col2:
                        st.markdown('<div class="section-header">Kết quả phân tích</div>', unsafe_allow_html=True)
                        
                        # Hiển thị kết quả với thiết kế đẹp hơn
                        if prediction == 1:
                            st.markdown("""
                            <div class="result-card tb">
                                <div class="result-icon">⚠️</div>
                                <div class="result-text">Phát hiện dấu hiệu lao phổi</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="result-card normal">
                                <div class="result-icon">✅</div>
                                <div class="result-text">Không phát hiện dấu hiệu lao phổi</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Hiển thị ảnh CAM
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(cam_image, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="image-caption">Vùng phân tích (CAM)</div>', unsafe_allow_html=True)
                        
                        # Hiển thị thông tin chi tiết trong card
                        st.markdown("""
                        <div class="info-card">
                            <div class="info-card-header">Thông tin chi tiết</div>
                            <div class="info-card-content">
                        """, unsafe_allow_html=True)
                        
                        # Tạo biểu đồ xác suất
                        fig, ax = plt.subplots(figsize=(8, 3))
                        labels = ['Bình thường', 'Lao phổi']
                        values = [prob_normal, prob_tb]
                        colors = ['#2ecc71', '#e74c3c']
                        
                        bars = ax.bar(labels, values, color=colors)
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Xác suất')
                        ax.set_title('Phân tích xác suất')
                        
                        # Thêm nhãn phần trăm trên mỗi cột
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                    f'{height:.2%}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                        
                        # Thêm thông tin thời gian xử lý
                        st.markdown(f"""
                            <div class="info-item">
                                <div class="info-label">⏱️ Thời gian xử lý:</div>
                                <div class="info-value">{process_time:.2f} giây</div>
                            </div>
                        """, unsafe_allow_html=True)

                        
                        if notes:
                            st.markdown(f"""
                                <div class="info-item">
                                    <div class="info-label">📝 Ghi chú:</div>
                                    <div class="info-value">{notes}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        # Tạo báo cáo PDF
                        pdf_buffer = create_pdf_report(
                            image, cam_image, prediction, prob_normal, prob_tb, process_time, 
                            uploaded_file.name, patient_info
                        )
                        
                        # Nút tải xuống PDF
                        st.download_button(
                            label="📄 Tải báo cáo PDF",
                            data=pdf_buffer,
                            file_name = f"bao_cao_xquang_{uploaded_file.name.split('.')[0]}.pdf",
                            mime="application/pdf",
                            key="download_single_pdf",
                            use_container_width=True
                        )

with tabs[1]:
    st.markdown("""
    <div class="tab-description">
        Tải lên nhiều ảnh X-quang phổi để phân tích hàng loạt và so sánh kết quả.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">Tải lên nhiều ảnh X-quang</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Kéo thả hoặc nhấp để chọn nhiều ảnh X-quang", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True, 
            key="multiple_upload",
            help="Hỗ trợ các định dạng: JPG, JPEG, PNG"
        )
        
        if uploaded_files:
            st.markdown(f"<div class='upload-info'>Đã tải lên {len(uploaded_files)} ảnh</div>", unsafe_allow_html=True)
            
            # Hiển thị ảnh đã tải lên dưới dạng lưới
            st.markdown("<div class='image-grid'>", unsafe_allow_html=True)
            for i, uploaded_file in enumerate(uploaded_files[:6]):  # Giới hạn hiển thị 6 ảnh
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, width=150, caption=uploaded_file.name)
            
            if len(uploaded_files) > 6:
                st.markdown(f"<div class='more-images'>+{len(uploaded_files) - 6} ảnh khác</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Ghi chú chung (không bắt buộc)
            batch_notes = st.text_area("Ghi chú chung (không bắt buộc):", key="batch_notes", height=100)
            
            col1_buttons = st.columns(2)
            with col1_buttons[0]:
                analyze_button = st.button("🔍 Phân tích tất cả", key="analyze_multiple", use_container_width=True)
            with col1_buttons[1]:
                reset_button = st.button("🔄 Làm mới", key="reset_multiple", use_container_width=True)
            
            if reset_button:
                st.session_state.needs_rerun = True
            
            if analyze_button:
                results = []
                images = []
                cam_images = []
                predictions = []
                probs_normal = []
                probs_tb = []
                process_times = []
                filenames = []
                pdf_buffers = []
                analysis_ids = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Đang phân tích ảnh {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Hiển thị ảnh gốc
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # Lưu ảnh tạm thời
                    temp_dir = tempfile.mkdtemp()
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    image.save(temp_file_path)
                    
                    # Phân tích ảnh
                    cam_image, prediction, prob_normal, prob_tb, process_time = apply_cam(
                        temp_file_path, model, preprocess, last_conv_layer)
                    
                    # Lưu kết quả vào database
                    analysis_id = save_analysis_to_db(
                        image, cam_image, prediction, prob_normal, prob_tb, process_time, 
                        uploaded_file.name, None, batch_notes
                    )
                    
                    # Tạo báo cáo PDF
                    pdf_buffer = create_pdf_report(
                        image, cam_image, prediction, prob_normal, prob_tb, process_time, uploaded_file.name)
                    
                    # Lưu kết quả
                    images.append(image)
                    cam_images.append(cam_image)
                    predictions.append(prediction)
                    probs_normal.append(prob_normal)
                    probs_tb.append(prob_tb)
                    process_times.append(process_time)
                    filenames.append(uploaded_file.name)
                    pdf_buffers.append(pdf_buffer)
                    analysis_ids.append(analysis_id)
                    
                    results.append({
                        'Tên file': uploaded_file.name,
                        'Kết quả': 'Lao phổi' if prediction == 1 else 'Bình thường',
                        'Xác suất bình thường': f'{prob_normal:.2%}',
                        'Xác suất lao phổi': f'{prob_tb:.2%}',
                        'Thời gian xử lý': f'{process_time:.2f}s'
                    })
                    
                    # Cập nhật thanh tiến trình
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Phân tích hoàn tất!")
                
                with col2:
                    st.markdown('<div class="section-header">Kết quả phân tích</div>', unsafe_allow_html=True)
                    
                    # Thống kê tổng quan
                    total = len(predictions)
                    normal_count = predictions.count(0)
                    tb_count = predictions.count(1)
                    
                    st.markdown(f"""
                    <div class="stats-container">
                        <div class="stat-card">
                            <div class="stat-value">{total}</div>
                            <div class="stat-label">Tổng số ảnh</div>
                        </div>
                        <div class="stat-card normal">
                            <div class="stat-value">{normal_count}</div>
                            <div class="stat-label">Bình thường</div>
                        </div>
                        <div class="stat-card tb">
                            <div class="stat-value">{tb_count}</div>
                            <div class="stat-label">Nghi ngờ lao phổi</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tạo biểu đồ tròn
                    if total > 0:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        labels = ['Bình thường', 'Lao phổi']
                        sizes = [normal_count, tb_count]
                        colors = ['#2ecc71', '#e74c3c']
                        explode = (0.1, 0.1)
                        
                        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                              shadow=True, startangle=90)
                        ax.axis('equal')
                        ax.set_title('Phân bố kết quả')
                        
                        st.pyplot(fig)
                    
                    # Hiển thị bảng kết quả
                    st.markdown("<div class='table-container'>", unsafe_allow_html=True)
                    st.dataframe(
                        df,
                        column_config={
                            "Kết quả": st.column_config.TextColumn(
                                "Kết quả",
                                help="Kết quả phân tích",
                                width="medium",
                            ),
                            "Xác suất bình thường": st.column_config.TextColumn(
                                "Xác suất bình thường",
                                width="medium",
                            ),
                            "Xác suất lao phổi": st.column_config.TextColumn(
                                "Xác suất lao phổi",
                                width="medium",
                            ),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Hiển thị ảnh với CAM cho các trường hợp lao phổi
                    if tb_count > 0:
                        st.markdown('<div class="section-header">Các trường hợp nghi ngờ lao phổi</div>', unsafe_allow_html=True)
                        st.markdown("<div class='cam-grid'>", unsafe_allow_html=True)
                        
                        for idx, (img, cam_img, pred, filename) in enumerate(zip(images, cam_images, predictions, filenames)):
                            if pred == 1:
                                col_img, col_cam = st.columns(2)
                                with col_img:
                                    st.image(img, caption=f"Gốc: {filename}", use_container_width=True)
                                with col_cam:
                                    st.image(cam_img, caption=f"CAM: {filename}", use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Tạo báo cáo PDF tổng hợp cho tất cả các ảnh
                    st.markdown('<div class="section-header">Báo cáo PDF</div>', unsafe_allow_html=True)
                    
                    # Tạo các nút tải xuống cho từng ảnh
                    for idx, (filename, pdf_buffer) in enumerate(zip(filenames, pdf_buffers)):
                        col_pdf = st.columns([3, 1])
                        with col_pdf[0]:
                            result_text = "Lao phổi" if predictions[idx] == 1 else "Bình thường"
                            result_class = "tb" if predictions[idx] == 1 else "normal"
                            st.markdown(f"""
                            <div class="pdf-item">
                                <div class="pdf-filename">{filename}</div>
                                <div class="pdf-result {result_class}">{result_text}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_pdf[1]:
                            st.download_button(
                                label="📄 Tải PDF",
                                data=pdf_buffer,
                                file_name=f"bao_cao_xquang_{filename.split('.')[ utrud: f"bao_cao_xquang_{filename.split('.')[0]}.pdf",
                                mime="application/pdf",
                                key=f"download_pdf_{idx}",
                                use_container_width=True
                            )

with tabs[2]:
    st.markdown("""
    <div class="tab-description">
        Thông tin về ứng dụng phát hiện lao phổi qua ảnh X-quang.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Giới thiệu</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Ứng dụng này sử dụng trí tuệ nhân tạo để phân tích ảnh X-quang phổi và phát hiện dấu hiệu của bệnh lao phổi. 
    Hệ thống được phát triển dựa trên mô hình học sâu (deep learning) được huấn luyện trên hàng nghìn ảnh X-quang 
    đã được bác sĩ chuyên khoa chẩn đoán.
    
    ### Cách sử dụng
    
    1. **Phân tích một ảnh**: Tải lên một ảnh X-quang phổi để phân tích. Bạn có thể chọn bệnh nhân từ danh sách 
       hoặc phân tích ảnh mà không cần liên kết với bệnh nhân cụ thể.
    
    2. **Phân tích nhiều ảnh**: Tải lên nhiều ảnh X-quang phổi để phân tích hàng loạt. Hệ thống sẽ xử lý tất cả 
       các ảnh và cung cấp kết quả tổng hợp.
    
    ### Lưu ý quan trọng
    
    - Kết quả phân tích từ ứng dụng này chỉ mang tính chất tham khảo và hỗ trợ.
    - Chẩn đoán cuối cùng nên được thực hiện bởi bác sĩ chuyên khoa có kinh nghiệm.
    - Độ chính xác của mô hình phụ thuộc vào chất lượng ảnh X-quang đầu vào.
    
    ### Về bệnh lao phổi
    
    Lao phổi là một bệnh truyền nhiễm do vi khuẩn Mycobacterium tuberculosis gây ra, chủ yếu ảnh hưởng đến phổi. 
    Bệnh lao phổi có thể được chẩn đoán thông qua nhiều phương pháp, trong đó X-quang ngực là một công cụ sàng lọc 
    quan trọng để phát hiện các tổn thương nghi ngờ.
    
    Các dấu hiệu của lao phổi trên phim X-quang có thể bao gồm:
    - Các đám mờ hoặc nốt ở phần trên của phổi
    - Hang lao (các khoang trống trong phổi)
    - Tổn thương dạng xơ hóa
    - Tràn dịch màng phổi
    
    ### Liên hệ
    
    Nếu bạn có bất kỳ câu hỏi hoặc phản hồi nào về ứng dụng, vui lòng liên hệ với chúng tôi qua email: support@example.com
    """)

if __name__ == "__main__":
    pass