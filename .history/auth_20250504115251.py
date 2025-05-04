import streamlit as st
import time
from jose import JWTError, jwt
from database import User, get_db, verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM
import sqlite3

# Kiểm tra trạng thái trang
if "page" not in st.session_state:
    st.session_state.page = "login"

# Chuyển hướng dựa trên trạng thái
if st.session_state.page == "admin":
    import pages.admin as admin
    admin.admin_page()
    st.stop()
elif st.session_state.page == "app":
    import app
    app.main()
    st.stop()

def login_page():
    st.set_page_config(
        page_title="Đăng nhập - Hệ thống phát hiện lao phổi",
        layout="centered",
        initial_sidebar_state="collapsed",
        page_icon="🫁"
    )
    
    # Import CSS từ file riêng
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="header">
        <div class="logo-container">
            <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" class="logo">
        </div>
        <div class="title-container">
            <h1>HỆ THỐNG PHÁT HIỆN LAO PHỔI</h1>
            <p class="subtitle">Đăng nhập để tiếp tục</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Kiểm tra nếu đã đăng nhập
    if "user_info" in st.session_state:
        if st.session_state.user_info["is_admin"]:
            st.success(f"Bạn đã đăng nhập với tư cách quản trị viên: {st.session_state.user_info['username']}")
            # st.button("Đi đến trang quản trị", on_click=lambda: st.switch_page("admin.py"))
            if st.button("Đi đến trang quản trị"):
                st.session_state.page = "admin"
                st.rerun()
        else:
            st.success(f"Bạn đã đăng nhập với tư cách bác sĩ: {st.session_state.user_info['username']}")
            # st.button("Đi đến trang chính", on_click=lambda: st.switch_page("app.py"))
            if st.button("Đi đến trang chính"):
                st.session_state.page = "app"
                st.rerun()
        
        if st.button("Đăng xuất"):
            del st.session_state.user_info
            st.rerun()
        return
    
    # Tab đăng nhập và đăng ký
    tab1, tab2 = st.tabs(["Đăng nhập", "Đăng ký"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Đăng nhập")
            username = st.text_input("Tên đăng nhập")
            password = st.text_input("Mật khẩu", type="password")
            submit = st.form_submit_button("Đăng nhập")
            
            if submit:
                if not username or not password:
                    st.error("Vui lòng nhập đầy đủ thông tin")
                    return
                
                # Kiểm tra đăng nhập
                db = next(get_db())
                user = db.query(User).filter(User.username == username).first()
                
                if not user or not verify_password(password, user.hashed_password):
                    st.error("Tên đăng nhập hoặc mật khẩu không chính xác")
                    return
                
                # Tạo token và lưu thông tin người dùng
                access_token = create_access_token(data={"sub": user.username})
                st.session_state.user_info = {
                    "id": user.id,
                    "username": user.username,
                    "full_name": user.full_name,
                    "is_admin": user.is_admin,
                    "token": access_token
                }
                
                with st.spinner("Đang đăng nhập..."):
                    time.sleep(1)
                
                if user.is_admin:
                    st.success("Đăng nhập thành công với tư cách quản trị viên!")
                    st.rerun()
                else:
                    st.success("Đăng nhập thành công!")
                    st.rerun()
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Đăng ký tài khoản")
            st.info("Chỉ quản trị viên mới có thể tạo tài khoản mới. Vui lòng liên hệ quản trị viên để được hỗ trợ.")
            new_username = st.text_input("Tên đăng nhập", key="new_username", disabled=True)
            new_email = st.text_input("Email", key="new_email", disabled=True)
            new_full_name = st.text_input("Họ và tên", key="new_full_name", disabled=True)
            new_password = st.text_input("Mật khẩu", type="password", key="new_password", disabled=True)
            confirm_password = st.text_input("Xác nhận mật khẩu", type="password", key="confirm_password", disabled=True)
            
            submit_register = st.form_submit_button("Đăng ký", disabled=True)

if __name__ == "__main__":
    login_page()
