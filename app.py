import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 实时口罩检测")

# 加载模型
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("请将 best.pt 文件放在项目根目录")
        st.stop()
    return YOLO(model_path)

# 初始化session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

model = load_model()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25)
    
    if st.button("🎥 开始实时检测"):
        st.session_state.running = True
        st.session_state.cap = cv2.VideoCapture(0)
    
    if st.button("⏹️ 停止"):
        st.session_state.running = False
        if st.session_state.cap:
            st.session_state.cap.release()

# 主显示区域
frame_placeholder = st.empty()
stats_placeholder = st.empty()

# 实时检测循环
if st.session_state.running and st.session_state.cap:
    ret, frame = st.session_state.cap.read()
    if ret:
        # YOLO检测
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        # 显示
        frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
        
        # 显示统计
        count = len(results[0].boxes) if results[0].boxes else 0
        stats_placeholder.info(f"检测目标数: {count}")
        
        # 继续下一帧
        time.sleep(0.03)
        st.rerun()
    else:
        st.session_state.running = False
        st.error("视频流结束")
