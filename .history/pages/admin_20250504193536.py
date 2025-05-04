import streamlit as st
import pandas as pd
import time
from database import User, Patient, Analysis, get_db, get_password_hash
import os
from datetime import datetime

def admin_page():
    st.set_page_config(
        page_title="Quản trị - Hệ thống phát hiện lao phổi",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🫁"
    )
    
    # Import CSS từ file riêng
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Kiểm tra đăng nhập
    if "user_info" not in st.session_state or not st.session_state.user_info["is_admin"]:
        st.warning("Bạn cần đăng nhập với tư cách quản trị viên để truy cập trang này")
        st.button("Đi đến trang đăng nhập", on_click=lambda: st.switch_page("auth.py"))
        return
    
    # Header
    st.markdown("""
    <div class="header">
        <div class="logo-container">
            <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" class="logo">
        </div>
        <div class="title-container">
            <h1>QUẢN TRỊ HỆ THỐNG</h1>
            <p class="subtitle">Quản lý người dùng và dữ liệu</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"**Xin chào, {st.session_state.user_info['full_name']}!**")
        st.markdown("---")
        
        menu = st.radio(
            "Chọn chức năng:",
            ["Quản lý tài khoản", "Quản lý bệnh nhân", "Thống kê hệ thống"]
        )
        
        st.markdown("---")
        if st.button("Đăng xuất"):
            del st.session_state.user_info
            st.switch_page("auth.py")
    
    # Main content
    if menu == "Quản lý tài khoản":
        manage_users()
    elif menu == "Quản lý bệnh nhân":
        manage_patients()
    else:
        system_statistics()

    # Kiểm tra làm mới gián tiếp
    if st.session_state.get("needs_rerun", False):
        st.session_state.needs_rerun = False
        st.rerun()

def manage_users():
    st.markdown('<div class="section-header">Quản lý tài khoản người dùng</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Danh sách người dùng", "Thêm người dùng mới"])
    
    with tab1:
        # Lấy danh sách người dùng từ database
        db = next(get_db())
        users = db.query(User).all()
        
        # Chuyển đổi thành DataFrame để hiển thị
        users_data = []
        for user in users:
            users_data.append({
                "ID": user.id,
                "Tên đăng nhập": user.username,
                "Email": user.email,
                "Họ và tên": user.full_name,
                "Vai trò": "Quản trị viên" if user.is_admin else "Bác sĩ",
                "Ngày tạo": user.created_at.strftime("%d/%m/%Y %H:%M")
            })
        
        df = pd.DataFrame(users_data)
        st.dataframe(df, use_container_width=True)
        
        # # Xóa người dùng
        # st.markdown("### Xóa người dùng")
        # col1, col2 = st.columns([3, 1])
        # with col1:
        #     user_to_delete = st.selectbox(
        #         "Chọn người dùng cần xóa:",
        #         options=[user.username for user in users if not user.is_admin],
        #         format_func=lambda x: f"{x} ({next((u.full_name for u in users if u.username == x), '')})"
        #     )
        
        # with col2:
        #     if st.button("Xóa người dùng", key="delete_user"):
        #         if user_to_delete:
        #             user = db.query(User).filter(User.username == user_to_delete).first()
        #             if user and not user.is_admin:
        #                 db.delete(user)
        #                 db.commit()
        #                 st.success(f"Đã xóa người dùng: {user_to_delete}")
        #                 time.sleep(1)
        #                 st.session_state.needs_rerun = True
        # Xóa người dùng
        st.markdown("### Xóa người dùng")
        col1, col2 = st.columns([3, 1])
        with col1:
            user_to_delete = st.selectbox(
                "Chọn người dùng cần xóa:",
                options=[user.username for user in users if not user.is_admin],
                format_func=lambda x: f"{x} ({next((u.full_name for u in users if u.username == x), '')})"
            )

        with col2:
            # Thêm CSS cho nút
            st.markdown(
                """
                <style>
                button[kind="primary"][key="delete_user"] {
                    margin-top: 35px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            if st.button("Xóa người dùng", key="delete_user"):
                if user_to_delete:
                    user = db.query(User).filter(User.username == user_to_delete).first()
                    if user and not user.is_admin:
                        db.delete(user)
                        db.commit()
                        st.success(f"Đã xóa người dùng: {user_to_delete}")
                        time.sleep(1)
                        st.session_state.needs_rerun = True
    
    with tab2:
        with st.form("add_user_form"):
            st.subheader("Thêm người dùng mới")
            
            new_username = st.text_input("Tên đăng nhập")
            new_email = st.text_input("Email")
            new_full_name = st.text_input("Họ và tên")
            new_password = st.text_input("Mật khẩu", type="password")
            confirm_password = st.text_input("Xác nhận mật khẩu", type="password")
            
            submit = st.form_submit_button("Thêm người dùng")
            
            if submit:
                if not new_username or not new_email or not new_full_name or not new_password:
                    st.error("Vui lòng nhập đầy đủ thông tin")
                    return
                
                if new_password != confirm_password:
                    st.error("Mật khẩu xác nhận không khớp")
                    return
                
                # Kiểm tra username và email đã tồn tại chưa
                db = next(get_db())
                existing_user = db.query(User).filter(User.username == new_username).first()
                if existing_user:
                    st.error(f"Tên đăng nhập '{new_username}' đã tồn tại")
                    return
                
                existing_email = db.query(User).filter(User.email == new_email).first()
                if existing_email:
                    st.error(f"Email '{new_email}' đã được sử dụng")
                    return
                
                # Tạo người dùng mới
                hashed_password = get_password_hash(new_password)
                new_user = User(
                    username=new_username,
                    email=new_email,
                    full_name=new_full_name,
                    hashed_password=hashed_password,
                    is_admin=False  # Mặc định là bác sĩ
                )
                
                db.add(new_user)
                db.commit()
                
                st.success(f"Đã thêm người dùng mới: {new_username}")

def manage_patients():
    st.markdown('<div class="section-header">Quản lý bệnh nhân</div>', unsafe_allow_html=True)
    
    # Lấy danh sách bệnh nhân từ database
    db = next(get_db())
    patients = db.query(Patient).all()
    
    # Chuyển đổi thành DataFrame để hiển thị
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
            "Địa chỉ": patient.address,
            "Số điện thoại": patient.phone,
            "Bác sĩ phụ trách": doctor_name,
            "Ngày tạo": patient.created_at.strftime("%d/%m/%Y")
        })
    
    df = pd.DataFrame(patients_data)
    st.dataframe(df, use_container_width=True)
    
    # Tìm kiếm bệnh nhân
    st.markdown("### Tìm kiếm bệnh nhân")
    search_term = st.text_input("Nhập tên hoặc mã bệnh nhân:")
    
    if search_term:
        filtered_patients = [p for p in patients_data if search_term.lower() in p["Họ và tên"].lower() or search_term.lower() in p["Mã bệnh nhân"].lower()]
        if filtered_patients:
            st.dataframe(pd.DataFrame(filtered_patients), use_container_width=True)
        else:
            st.info("Không tìm thấy bệnh nhân phù hợp")
    
    # Xóa bệnh nhân
    st.markdown("### Xóa bệnh nhân")
    col1, col2 = st.columns([3, 1])
    with col1:
        patient_to_delete = st.selectbox(
            "Chọn bệnh nhân cần xóa:",
            options=[p.id for p in patients],
            format_func=lambda x: f"{next((p.patient_code for p in patients if p.id == x), '')} - {next((p.full_name for p in patients if p.id == x), '')}"
        )
    
    with col2:
        if st.button("Xóa bệnh nhân", key="delete_patient"):
            if patient_to_delete:
                patient = db.query(Patient).filter(Patient.id == patient_to_delete).first()
                if patient:
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
                    st.session_state.needs_rerun = True

def system_statistics():
    st.markdown('<div class="section-header">Thống kê hệ thống</div>', unsafe_allow_html=True)
    
    db = next(get_db())
    
    # Đếm số lượng
    user_count = db.query(User).count()
    doctor_count = db.query(User).filter(User.is_admin == False).count()
    patient_count = db.query(Patient).count()
    analysis_count = db.query(Analysis).count()
    
    # Hiển thị thống kê
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">Người dùng</div>
        </div>
        """.format(user_count), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">Bác sĩ</div>
        </div>
        """.format(doctor_count), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">Bệnh nhân</div>
        </div>
        """.format(patient_count), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">Phân tích</div>
        </div>
        """.format(analysis_count), unsafe_allow_html=True)
    
    # Thống kê phân tích theo thời gian
    st.markdown("### Phân tích theo thời gian")
    
    # Lấy dữ liệu phân tích theo ngày
    analyses = db.query(Analysis).all()
    analysis_dates = [a.created_at.date() for a in analyses]
    analysis_results = [a.prediction for a in analyses]
    
    if analyses:
        # Tạo DataFrame để vẽ biểu đồ
        date_df = pd.DataFrame({
            'date': analysis_dates,
            'result': analysis_results
        })
        
        # Đếm số lượng phân tích theo ngày và kết quả
        date_counts = date_df.groupby(['date', 'result']).size().reset_index(name='count')
        date_counts['result'] = date_counts['result'].map({0: 'Bình thường', 1: 'Lao phổi'})
        
        # Vẽ biểu đồ
        st.bar_chart(date_counts.pivot(index='date', columns='result', values='count').fillna(0))
    else:
        st.info("Chưa có dữ liệu phân tích")
    
    # Thống kê tỷ lệ phát hiện
    st.markdown("### Tỷ lệ phát hiện lao phổi")
    
    if analyses:
        normal_count = sum(1 for a in analyses if a.prediction == 0)
        tb_count = sum(1 for a in analyses if a.prediction == 1)
        
        data = pd.DataFrame({
            'Kết quả': ['Bình thường', 'Lao phổi'],
            'Số lượng': [normal_count, tb_count]
        })
        
        st.bar_chart(data.set_index('Kết quả'))
    else:
        st.info("Chưa có dữ liệu phân tích")

if __name__ == "__main__":
    admin_page()