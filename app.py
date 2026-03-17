import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 实时口罩检测")
st.markdown("自动连续拍照实现实时检测")

# 加载模型
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("请将 best.pt 文件放在项目根目录")
        st.stop()
    return YOLO(model_path)

model = load_model()

# 初始化session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_image' not in st.session_state:
    st.session_state.last_image = None
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 设置")
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25)
    capture_interval = st.slider("拍照间隔(秒)", 0.1, 1.0, 0.3)
    
    st.markdown("---")
    st.markdown("### 控制")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ 开始实时", use_container_width=True):
            st.session_state.running = True
    with col2:
        if st.button("⏹️ 停止", use_container_width=True):
            st.session_state.running = False
    
    st.markdown("---")
    st.markdown("### 检测统计")
    stats_placeholder = st.empty()

# 主界面
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 摄像头画面")
    camera_input = st.camera_input("拍照", key="camera", disabled=not st.session_state.running, label_visibility="collapsed")

with col2:
    st.markdown("### 检测结果")
    result_placeholder = st.empty()

# 处理每一帧
if camera_input is not None and st.session_state.running:
    # 读取图像
    bytes_data = camera_input.getvalue()
    img_array = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # YOLO检测
    results = model(img, conf=conf_threshold, verbose=False)
    annotated_img = results[0].plot()
    
    # 保存到session state
    st.session_state.last_image = annotated_img
    st.session_state.detection_count = len(results[0].boxes) if results[0].boxes else 0
    
    # 显示结果
    result_placeholder.image(annotated_img, channels="BGR", use_column_width=True)
    
    # 更新统计
    stats_placeholder.metric("当前检测目标数", st.session_state.detection_count)
    
    # 自动重新运行以实现连续检测
    if st.session_state.running:
        time.sleep(capture_interval)
        st.rerun()

elif st.session_state.running:
    st.info("等待摄像头启动...")

# 显示上一次的结果
if not st.session_state.running and st.session_state.last_image is not None:
    result_placeholder.image(st.session_state.last_image, channels="BGR", use_column_width=True)
