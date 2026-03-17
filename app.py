import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 摄像头实时口罩检测")
st.markdown("打开摄像头，实时显示检测结果")

# 加载模型
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("请将 best.pt 文件放在项目根目录")
        st.stop()
    return YOLO(model_path)

model = load_model()
st.success("✅ 模型加载成功")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 设置")
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25)
    camera_id = st.number_input("摄像头ID", min_value=0, max_value=10, value=0)
    
    st.markdown("---")
    if st.button("🎥 开始检测"):
        st.session_state['detect'] = True
    if st.button("⏹️ 停止检测"):
        st.session_state['detect'] = False

# 初始化
if 'detect' not in st.session_state:
    st.session_state['detect'] = False

# 视频显示区域
frame_placeholder = st.empty()
status_text = st.empty()

# 实时检测
if st.session_state['detect']:
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        st.error("无法打开摄像头")
        st.session_state['detect'] = False
    else:
        status_text.success("✅ 实时检测中...")
        
        while st.session_state['detect']:
            ret, frame = cap.read()
            if not ret:
                st.error("无法读取视频帧")
                break
            
            # YOLO检测
            results = model(frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # 显示结果
            frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
        
        # 释放摄像头
        cap.release()
        status_text.info("⏸️ 检测已停止")
else:
    frame_placeholder.empty()
    status_text.info("点击左侧'开始检测'按钮启动摄像头")
