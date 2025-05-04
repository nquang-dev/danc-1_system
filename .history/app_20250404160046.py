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


# Thiết lập trang
st.set_page_config(
    page_title="Phát hiện lao phổi qua X-quang", 
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🫁"
)


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
def create_pdf_report(image, cam_image, prediction, prob_normal, prob_tb, process_time, filename=None):
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
            
            col1_buttons = st.columns(2)
            with col1_buttons[0]:
                analyze_button = st.button("🔍 Phân tích ảnh", key="analyze_single", use_container_width=True)
            with col1_buttons[1]:
                reset_button = st.button("🔄 Làm mới", key="reset_single", use_container_width=True)
            
            if reset_button:
                st.rerun()
            
            if analyze_button:
                with st.spinner("⏳ Đang phân tích ảnh..."):
                    # Lưu ảnh tạm thời
                    temp_dir = tempfile.mkdtemp()
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    image.save(temp_file_path)
                    
                    # Phân tích ảnh
                    cam_image, prediction, prob_normal, prob_tb, process_time = apply_cam(
                        temp_file_path, model, preprocess, last_conv_layer)
                    
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
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        # Tạo báo cáo PDF
                        pdf_buffer = create_pdf_report(
                            image, cam_image, prediction, prob_normal, prob_tb, process_time, uploaded_file.name)
                        
                        # Nút tải xuống PDF
                        st.download_button(
                            label="📄 Tải báo cáo PDF",
                            data=pdf_buffer,
                            file_name=f"bao_cao_xquang_{uploaded_file.name.split('.')[0]}.pdf",
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
            
            col1_buttons = st.columns(2)
            with col1_buttons[0]:
                analyze_button = st.button("🔍 Phân tích tất cả", key="analyze_multiple", use_container_width=True)
            with col1_buttons[1]:
                reset_button = st.button("🔄 Làm mới", key="reset_multiple", use_container_width=True)
            
            if reset_button:
                st.rerun()
            
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
                    
                    # Hiển thị bảng kết quả với thiết kế đẹp hơn
                    df = pd.DataFrame(results)
                    
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
                                file_name=f"bao_cao_xquang_{filename.split('.')[0]}.pdf",
                                mime="application/pdf",
                                key=f"download_pdf_{idx}",
                                use_container_width=True
                            )

with tabs[2]:
    st.title("Thông tin về ứng dụng")
    
    st.markdown("**Ứng dụng phát hiện lao phổi qua ảnh X-quang** sử dụng trí tuệ nhân tạo giúp phát hiện sớm các dấu hiệu của bệnh lao phổi thông qua phân tích ảnh X-quang ngực.")
    
    # Phần Cách sử dụng
    st.subheader("Cách sử dụng")
    st.markdown("""
    1. Chọn tab "Phân tích một ảnh" hoặc "Phân tích nhiều ảnh" tùy theo nhu cầu sử dụng
    2. Tải lên ảnh X-quang phổi cần phân tích (định dạng JPG, JPEG hoặc PNG)
    3. Nhấn nút "Phân tích ảnh" để bắt đầu quá trình phân tích
    4. Xem kết quả và tải xuống báo cáo PDF nếu cần
    """)
    
    # Phần Về mô hình AI
    st.subheader("Về mô hình AI")
    st.info("""
    Ứng dụng sử dụng mô hình học sâu (Deep Learning) được huấn luyện trên tập dữ liệu X-quang phổi lớn để phân biệt giữa ảnh X-quang bình thường và ảnh có dấu hiệu lao phổi.
    
    Công nghệ Class Activation Map (CAM) được sử dụng để trực quan hóa vùng nghi ngờ trên ảnh X-quang, giúp bác sĩ có thêm thông tin trong quá trình chẩn đoán.
    """)
    
    # Phần Lưu ý quan trọng với màu nổi bật
    st.subheader("Lưu ý quan trọng")
    st.error("""
    **Kết quả từ ứng dụng này chỉ mang tính chất tham khảo và hỗ trợ.**
    
    Không nên sử dụng kết quả từ ứng dụng này để thay thế chẩn đoán của bác sĩ chuyên khoa. Vui lòng tham khảo ý kiến của bác sĩ để có chẩn đoán chính xác và phương pháp điều trị phù hợp.
    """)
    
    # Phần Đội ngũ phát triển
    st.subheader("Đội ngũ phát triển")
    st.success("""
    Ứng dụng được phát triển như một phần của đồ án chuyên ngành về ứng dụng trí tuệ nhân tạo trong nhận dạng và xử lý ảnh y tế.
    
    © 2025 - Đồ án chuyên ngành 1 - Ứng dụng trí tuệ nhân tạo trong nhận dạng và xử lý ảnh
    """)
