import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 摄像头实时口罩检测")

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
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 设置")
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25)
    
    st.markdown("---")
    st.markdown("### 控制")
    
    if st.button("🎥 开始实时检测", use_container_width=True):
        st.session_state.running = True
        st.session_state.frame_count = 0
    
    if st.button("⏹️ 停止检测", use_container_width=True):
        st.session_state.running = False
    
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("1. 点击'开始'按钮")
    st.markdown("2. 允许摄像头权限")
    st.markdown("3. 自动连续拍照检测")
    st.markdown("4. 点击'停止'结束")

# 主界面
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📷 摄像头画面")
    # 每次刷新都会重新拍照
    camera_key = f"cam_{st.session_state.frame_count}"
    img_data = st.camera_input("拍照", key=camera_key, label_visibility="collapsed", disabled=not st.session_state.running)

with col2:
    st.markdown("### 🔍 检测结果")
    result_placeholder = st.empty()
    stats_placeholder = st.empty()

# 处理拍照的图像
if img_data is not None and st.session_state.running:
    # 增加帧计数
    st.session_state.frame_count += 1
    
    # 读取图像
    bytes_data = img_data.getvalue()
    img_array = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # YOLO检测
    with st.spinner("检测中..."):
        results = model(img, conf=conf_threshold, verbose=False)
        annotated_img = results[0].plot()
    
    # 保存结果
    st.session_state.last_result = annotated_img
    
    # 显示结果
    result_placeholder.image(annotated_img, channels="BGR", use_column_width=True)
    
    # 显示检测统计
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        stats_placeholder.metric("当前检测目标数", count)
        
        # 显示类别统计
        if count > 0:
            class_counts = {}
            for box in results[0].boxes:
                cls = int(box.cls[0].item())
                name = results[0].names[cls]
                class_counts[name] = class_counts.get(name, 0) + 1
            
            stats_placeholder.write("类别统计:")
            for name, cnt in class_counts.items():
                stats_placeholder.write(f"- {name}: {cnt}")
    else:
        stats_placeholder.info("未检测到目标")
    
    # 自动重新运行实现连续检测
    time.sleep(0.1)  # 小延迟
    st.rerun()

# 显示上次的结果
if not st.session_state.running and st.session_state.last_result is not None:
    result_placeholder.image(st.session_state.last_result, channels="BGR", use_column_width=True)
    stats_placeholder.info("检测已停止")

# 初始状态
if not st.session_state.running and st.session_state.last_result is None:
    result_placeholder.info("点击左侧'开始实时检测'按钮启动摄像头")
