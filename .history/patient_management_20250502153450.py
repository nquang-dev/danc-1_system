import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from database import User, Patient, Analysis, get_db
from PIL import Image
import io
import uuid

def patient_management():
    st.set_page_config(
        page_title="Quản lý bệnh nhân - Hệ thống phát hiện lao phổi",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🫁"
    )
    
    # Import CSS từ file riêng
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Kiểm tra đăng nhập
    if "user_info" not in st.session_state:
        st.warning("Bạn cần đăng nhập để truy cập trang này")
        st.button("Đi đến trang đăng nhập", on_click=lambda: st.switch_page("auth.py"))
        return
    
    # Header
    st.markdown("""
    <div class="header">
        <div class="logo-container">
            <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" class="logo">
        </div>
        <div class="title-container">
            <h1>QUẢN LÝ BỆNH NHÂN</h1>
            <p class="subtitle">Hồ sơ và lịch sử phân tích</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"**Xin chào, {st.session_state.user_info['full_name']}!**")
        st.markdown("---")
        
        menu = st.radio(
            "Chọn chức năng:",
            ["Danh sách bệnh nhân", "Thêm bệnh nhân mới", "Lịch sử phân tích"]
        )
        
        st.markdown("---")
        if st.button("Trang chính"):
            st.switch_page("app.py")
        
        if st.button("Đăng xuất"):
            del st.session_state.user_info
            st.rerun()
    
    # Main content
    if menu == "Danh sách bệnh nhân":
        list_patients()
    elif menu == "Thêm bệnh nhân mới":
        add_patient()
    else:
        analysis_history()

def list_patients():
    st.markdown('<div class="section-header">Danh sách bệnh nhân</div>', unsafe_allow_html=True)
    
    # Tìm kiếm
    search_term = st.text_input("Tìm kiếm bệnh nhân (theo tên hoặc mã):")
    
    # Lấy danh sách bệnh nhân từ database
    db = next(get_db())
    
    if st.session_state.user_info["is_admin"]:
        # Admin có thể xem tất cả bệnh nhân
        query = db.query(Patient)
    else:
        # Bác sĩ chỉ xem bệnh nhân của mình
        query = db.query(Patient).filter(Patient.doctor_id == st.session_state.user_info["id"])
    
    # Áp dụng tìm kiếm nếu có
    if search_term:
        query = query.filter(
            (Patient.full_name.like(f"%{search_term}%")) | 
            (Patient.patient_code.like(f"%{search_term}%"))
        )
    
    patients = query.all()
    
    # Hiển thị danh sách bệnh nhân
    if patients:
        patients_data = []
        for patient in patients:
            doctor_name = db.query(User.full_name).filter(User.id == patient.doctor_id).first()
            doctor_name = doctor_name[0] if doctor_name else "Không có"
            
            patients_data.append({
                "ID": patient.id,
                "Mã bệnh nhân": patient.patient_code,
                "Họ và tên": patient.full_name,
                "Tuổi": patient.age,
                "Giới tính": patient.gender,
                "Địa chỉ": patient.address or "",
                "Số điện thoại": patient.phone or "",
                "Bác sĩ phụ trách": doctor_name,
                "Ngày tạo": patient.created_at.strftime("%d/%m/%Y")
            })
        
        df = pd.DataFrame(patients_data)
        st.dataframe(df, use_container_width=True)
        
        # Chọn bệnh nhân để xem chi tiết
        selected_patient = st.selectbox(
            "Chọn bệnh nhân để xem chi tiết:",
            options=[p.id for p in patients],
            format_func=lambda x: f"{next((p.patient_code for p in patients if p.id == x), '')} - {next((p.full_name for p in patients if p.id == x), '')}"
        )
        
        if selected_patient:
            view_patient_details(selected_patient)
    else:
        st.info("Không tìm thấy bệnh nhân nào")

def view_patient_details(patient_id):
    db = next(get_db())
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    
    if not patient:
        st.error("Không tìm thấy thông tin bệnh nhân")
        return
    
    st.markdown(f"### Chi tiết bệnh nhân: {patient.full_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Mã bệnh nhân:** {patient.patient_code}")
        st.markdown(f"**Họ và tên:** {patient.full_name}")
        st.markdown(f"**Tuổi:** {patient.age}")
        st.markdown(f"**Giới tính:** {patient.gender}")
    
    with col2:
        st.markdown(f"**Địa chỉ:** {patient.address or 'Không có'}")
        st.markdown(f"**Số điện thoại:** {patient.phone or 'Không có'}")
        doctor = db.query(User).filter(User.id == patient.doctor_id).first()
        st.markdown(f"**Bác sĩ phụ trách:** {doctor.full_name if doctor else 'Không có'}")
        st.markdown(f"**Ngày tạo hồ sơ:** {patient.created_at.strftime('%d/%m/%Y %H:%M')}")
    
    # Hiển thị lịch sử phân tích
    st.markdown("### Lịch sử phân tích")
    
    analyses = db.query(Analysis).filter(Analysis.patient_id == patient.id).order_by(Analysis.created_at.desc()).all()
    
    if analyses:
        analyses_data = []
        for analysis in analyses:
            analyses_data.append({
                "ID": analysis.id,
                "Ngày phân tích": analysis.created_at.strftime("%d/%m/%Y %H:%M"),
                "Kết quả": "Lao phổi" if analysis.prediction == 1 else "Bình thường",
                "Xác suất bình thường": f"{analysis.probability_normal:.2%}",
                "Xác suất lao phổi": f"{analysis.probability_tb:.2%}",
                "Thời gian xử lý": f"{analysis.process_time:.2f}s",
                "Ghi chú": analysis.notes or ""
            })
        
        st.dataframe(pd.DataFrame(analyses_data), use_container_width=True)
        
        # Chọn phân tích để xem chi tiết
        selected_analysis = st.selectbox(
            "Chọn phân tích để xem chi tiết:",
            options=[a.id for a in analyses],
            format_func=lambda x: f"Phân tích ngày {next((a.created_at.strftime('%d/%m/%Y %H:%M') for a in analyses if a.id == x), '')}"
        )
        
        if selected_analysis:
            analysis = next((a for a in analyses if a.id == selected_analysis), None)
            if analysis:
                st.markdown(f"### Chi tiết phân tích #{analysis.id}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if analysis.image_path and os.path.exists(analysis.image_path):
                        st.image(analysis.image_path, caption="Ảnh X-quang gốc", use_column_width=True)
                    else:
                        st.info("Không tìm thấy ảnh gốc")
                
                with col2:
                    if analysis.cam_image_path and os.path.exists(analysis.cam_image_path):
                        st.image(analysis.cam_image_path, caption="Ảnh phân tích (CAM)", use_column_width=True)
                    else:
                        st.info("Không tìm thấy ảnh phân tích")
                
                # Hiển thị kết quả
                if analysis.prediction == 1:
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
                
                # Hiển thị thông tin chi tiết
                st.markdown("#### Thông tin chi tiết")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Xác suất bình thường:** {analysis.probability_normal:.2%}")
                    st.markdown(f"**Xác suất lao phổi:** {analysis.probability_tb:.2%}")
                with col2:
                    st.markdown(f"**Thời gian xử lý:** {analysis.process_time:.2f} giây")
                    doctor = db.query(User).filter(User.id == analysis.doctor_id).first()
                    st.markdown(f"**Bác sĩ phân tích:** {doctor.full_name if doctor else 'Không có'}")
                
                # Ghi chú
                if analysis.notes:
                    st.markdown(f"**Ghi chú:** {analysis.notes}")
                
                # Nút xóa phân tích
                if st.button("Xóa phân tích này", key=f"delete_analysis_{analysis.id}"):
                    # Xóa file ảnh nếu tồn tại
                    if analysis.image_path and os.path.exists(analysis.image_path):
                        os.remove(analysis.image_path)
                    if analysis.cam_image_path and os.path.exists(analysis.cam_image_path):
                        os.remove(analysis.cam_image_path)
                    
                    # Xóa từ database
                    db.delete(analysis)
                    db.commit()
                    st.success("Đã xóa phân tích")
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("Bệnh nhân này chưa có lịch sử phân tích")
    
    # Nút chỉnh sửa và xóa bệnh nhân
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Chỉnh sửa thông tin bệnh nhân", key=f"edit_patient_{patient.id}"):
            st.session_state.edit_patient_id = patient.id
            st.session_state.edit_patient_data = {
                "patient_code": patient.patient_code,
                "full_name": patient.full_name,
                "age": patient.age,
                "gender": patient.gender,
                "address": patient.address,
                "phone": patient.phone
            }
            st.rerun()
    
    with col2:
        if st.button("Xóa bệnh nhân này", key=f"delete_patient_{patient.id}"):
            # Xác nhận xóa
            confirm = st.checkbox("Tôi xác nhận muốn xóa bệnh nhân này và tất cả dữ liệu liên quan", key=f"confirm_delete_{patient.id}")
            if confirm:
                # Xóa các phân tích liên quan
                analyses = db.query(Analysis).filter(Analysis.patient_id == patient.id).all()
                for analysis in analyses:
                    # Xóa file ảnh nếu tồn tại
                    if analysis.image_path and os.path.exists(analysis.image_path):
                        os.remove(analysis.image_path)
                    if analysis.cam_image_path and os.path.exists(analysis.cam_image_path):
                        os.remove(analysis.cam_image_path)
                    db.delete(analysis)
                
                # Xóa bệnh nhân
                db.delete(patient)
                db.commit()
                st.success(f"Đã xóa bệnh nhân: {patient.full_name}")
                time.sleep(1)
                st.rerun()
    
    # Form chỉnh sửa bệnh nhân
    if "edit_patient_id" in st.session_state and st.session_state.edit_patient_id == patient.id:
        st.markdown("### Chỉnh sửa thông tin bệnh nhân")
        
        with st.form(key=f"edit_patient_form_{patient.id}"):
            patient_code = st.text_input("Mã bệnh nhân", value=st.session_state.edit_patient_data["patient_code"])
            full_name = st.text_input("Họ và tên", value=st.session_state.edit_patient_data["full_name"])
            age = st.number_input("Tuổi", min_value=0, max_value=120, value=st.session_state.edit_patient_data["age"])
            gender = st.selectbox("Giới tính", options=["Nam", "Nữ", "Khác"], index=["Nam", "Nữ", "Khác"].index(st.session_state.edit_patient_data["gender"]))
            address = st.text_input("Địa chỉ", value=st.session_state.edit_patient_data["address"] or "")
            phone = st.text_input("Số điện thoại", value=st.session_state.edit_patient_data["phone"] or "")
            
            submit = st.form_submit_button("Cập nhật thông tin")
            
            if submit:
                if not patient_code or not full_name or not age:
                    st.error("Vui lòng nhập đầy đủ thông tin bắt buộc")
                    return
                
                # Kiểm tra mã bệnh nhân đã tồn tại chưa (nếu thay đổi)
                if patient_code != patient.patient_code:
                    existing_code = db.query(Patient).filter(Patient.patient_code == patient_code).first()
                    if existing_code:
                        st.error(f"Mã bệnh nhân '{patient_code}' đã tồn tại")
                        return
                
                # Cập nhật thông tin
                patient.patient_code = patient_code
                patient.full_name = full_name
                patient.age = age
                patient.gender = gender
                patient.address = address
                patient.phone = phone
                
                db.commit()
                
                # Xóa session state
                del st.session_state.edit_patient_id
                del st.session_state.edit_patient_data
                
                st.success("Đã cập nhật thông tin bệnh nhân")
                time.sleep(1)
                st.rerun()

def add_patient():
    st.markdown('<div class="section-header">Thêm bệnh nhân mới</div>', unsafe_allow_html=True)
    
    with st.form("add_patient_form"):
        st.subheader("Thông tin bệnh nhân")
        
        # Tự động tạo mã bệnh nhân
        today = datetime.now().strftime("%Y%m%d")
        db = next(get_db())
        count = db.query(Patient).count()
        suggested_code = f"BN{today}{count+1:03d}"
        
        patient_code = st.text_input("Mã bệnh nhân", value=suggested_code)
        full_name = st.text_input("Họ và tên")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Tuổi", min_value=0, max_value=120, value=30)
        with col2:
            gender = st.selectbox("Giới tính", options=["Nam", "Nữ", "Khác"])
        
        address = st.text_input("Địa chỉ (không bắt buộc)")
        phone = st.text_input("Số điện thoại (không bắt buộc)")
        
        submit = st.form_submit_button("Thêm bệnh nhân")
        
        if submit:
            if not patient_code or not full_name or not age:
                st.error("Vui lòng nhập đầy đủ thông tin bắt buộc")
                return
            
            # Kiểm tra mã bệnh nhân đã tồn tại chưa
            db = next(get_db())
            existing_code = db.query(Patient).filter(Patient.patient_code == patient_code).first()
            if existing_code:
                st.error(f"Mã bệnh nhân '{patient_code}' đã tồn tại")
                return
            
            # Tạo bệnh nhân mới
            new_patient = Patient(
                patient_code=patient_code,
                full_name=full_name,
                age=age,
                gender=gender,
                address=address if address else None,
                phone=phone if phone else None,
                doctor_id=st.session_state.user_info["id"]
            )
            
            db.add(new_patient)
            db.commit()
            
            st.success(f"Đã thêm bệnh nhân mới: {full_name}")
            
            # Thêm nút để phân tích ngay cho bệnh nhân này
            if st.button("Phân tích X-quang cho bệnh nhân này"):
                st.session_state.selected_patient_id = new_patient.id
                st.switch_page("app.py")

def analysis_history():
    st.markdown('<div class="section-header">Lịch sử phân tích</div>', unsafe_allow_html=True)
    
    # Lấy danh sách phân tích từ database
    db = next(get_db())
    
    if st.session_state.user_info["is_admin"]:
        # Admin có thể xem tất cả phân tích
        analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
    else:
        # Bác sĩ chỉ xem phân tích của mình
        analyses = db.query(Analysis).filter(Analysis.doctor_id == st.session_state.user_info["id"]).order_by(Analysis.created_at.desc()).all()
    
    if analyses:
        analyses_data = []
        for analysis in analyses:
            patient = db.query(Patient).filter(Patient.id == analysis.patient_id).first()
            patient_name = patient.full_name if patient else "Không có"
            patient_code = patient.patient_code if patient else "Không có"
            
            analyses_data.append({
                "ID": analysis.id,
                "Ngày phân tích": analysis.created_at.strftime("%d/%m/%Y %H:%M"),
                "Mã bệnh nhân": patient_code,
                "Tên bệnh nhân": patient_name,
                "Kết quả": "Lao phổi" if analysis.prediction == 1 else "Bình thường",
                "Xác suất bình thường": f"{analysis.probability_normal:.2%}",
                "Xác suất lao phổi": f"{analysis.probability_tb:.2%}"
            })
        
        df = pd.DataFrame(analyses_data)
        st.dataframe(df, use_container_width=True)
        
        # Thống kê
        st.markdown("### Thống kê phân tích")
        
        normal_count = sum(1 for a in analyses if a.prediction == 0)
        tb_count = sum(1 for a in analyses if a.prediction == 1)
        total_count = len(analyses)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Tổng số phân tích</div>
            </div>
            """.format(total_count), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-card normal">
                <div class="stat-value">{}</div>
                <div class="stat-label">Bình thường</div>
            </div>
            """.format(normal_count), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-card tb">
                <div class="stat-value">{}</div>
                <div class="stat-label">Lao phổi</div>
            </div>
            """.format(tb_count), unsafe_allow_html=True)
        
        # Biểu đồ tỷ lệ
        data = pd.DataFrame({
            'Kết quả': ['Bình thường', 'Lao phổi'],
            'Số lượng': [normal_count, tb_count]
        })
        
        st.bar_chart(data.set_index('Kết quả'))
    else:
        st.info("Chưa có lịch sử phân tích")

if __name__ == "__main__":
    patient_management()

