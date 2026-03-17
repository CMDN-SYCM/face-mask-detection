import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

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
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 设置")
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25)
    fps = st.slider("检测帧率(FPS)", 1, 10, 5)
    
    st.markdown("---")
    st.markdown("### 控制")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ 开始", use_container_width=True):
            st.session_state.running = True
            st.session_state.frame_count = 0
    with col2:
        if st.button("⏹️ 停止", use_container_width=True):
            st.session_state.running = False
    
    st.markdown("---")
    st.markdown("### 检测统计")
    stats_placeholder = st.empty()

# 注入JavaScript实现自动点击
if st.session_state.running:
    st.markdown("""
    <script>
    // 自动点击拍照按钮
    function autoClick() {
        const captureBtn = document.querySelector('button[data-testid="baseButton-secondary"]');
        if (captureBtn) {
            captureBtn.click();
        }
    }
    
    // 设置定时器
    setInterval(autoClick, 200); // 每200ms点击一次
    </script>
    """, unsafe_allow_html=True)

# 主界面布局
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 摄像头画面")
    # 每次刷新都会重新创建camera_input
    camera_key = f"camera_{st.session_state.frame_count}"
    img_data = st.camera_input("拍照", key=camera_key, label_visibility="collapsed")

with col2:
    st.markdown("### 检测结果")
    result_placeholder = st.empty()

# 处理拍照的图像
if img_data is not None and st.session_state.running:
    st.session_state.frame_count += 1
    
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
    stats_placeholder.metric("当前检测目标数", count)
    
    # 控制帧率
    time.sleep(1/fps)
    
    # 重新运行实现连续检测
    st.rerun()

# 停止时显示最后结果
if not st.session_state.running and 'last_result' in st.session_state:
    result_placeholder.image(st.session_state.last_result, channels="BGR", use_column_width=True)
