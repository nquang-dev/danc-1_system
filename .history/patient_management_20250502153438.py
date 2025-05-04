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
            (Patient
