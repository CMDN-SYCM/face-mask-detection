import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 实时口罩检测")
st.markdown("使用摄像头实时检测")

# 加载模型
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("请将 best.pt 文件放在项目根目录")
        st.stop()
    return YOLO(model_path)

model = load_model()

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 设置")
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25)
    
    st.header("📹 摄像头")
    camera_id = st.number_input("摄像头ID", min_value=0, max_value=10, value=0)
    
    start = st.button("开始检测")
    stop = st.button("停止")

# 视频显示区域
frame_placeholder = st.empty()

if start:
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        st.error("无法打开摄像头")
        st.stop()
    
    st.success("摄像头已打开，正在实时检测...")
    
    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("无法读取视频帧")
            break
        
        # YOLO检测
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        # 转换为RGB显示
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # 显示结果
        frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
        
        # 小延迟，降低CPU使用
        time.sleep(0.03)
    
    cap.release()
    st.warning("检测已停止")
