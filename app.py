import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 摄像头实时口罩检测")
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
if 'count' not in st.session_state:
    st.session_state.count = 0

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 设置")
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25)
    refresh_rate = st.slider("刷新速度(毫秒)", 100, 500, 200)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎥 开始"):
            st.session_state.running = True
    with col2:
        if st.button("⏹️ 停止"):
            st.session_state.running = False
    
    st.markdown("---")
    stats_placeholder = st.empty()

# 自动刷新实现连续拍照
if st.session_state.running:
    st_autorefresh(interval=refresh_rate, key="camera_refresh")

# 主界面布局
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 摄像头画面")
    # 每次刷新都会重新拍照
    camera_key = f"camera_{st.session_state.count}"
    img_data = st.camera_input("拍照", key=camera_key, label_visibility="collapsed")

with col2:
    st.markdown("### 检测结果")
    result_placeholder = st.empty()

# 处理拍照的图像
if img_data is not None and st.session_state.running:
    st.session_state.count += 1
    
    # 读取图像
    bytes_data = img_data.getvalue()
    img_array = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # YOLO检测
    results = model(img, conf=conf_threshold, verbose=False)
    annotated_img = results[0].plot()
    
    # 显示结果
    result_placeholder.image(annotated_img, channels="BGR", use_column_width=True)
    
    # 更新统计
    count = len(results[0].boxes) if results[0].boxes else 0
    stats_placeholder.metric("检测目标数", count)

# 停止时显示提示
if not st.session_state.running:
    result_placeholder.info("点击'开始'按钮启动检测")
