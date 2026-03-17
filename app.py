import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 实时口罩检测")
st.markdown("点击拍照按钮进行检测")

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
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25, 0.05)
    
    if model and hasattr(model, 'names'):
        st.write("检测类别:", list(model.names.values()))
    
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("1. 允许摄像头权限")
    st.markdown("2. 点击拍照按钮")
    st.markdown("3. 查看检测结果")

# 摄像头输入
enable = st.checkbox("打开摄像头", value=True)
img_file_buffer = st.camera_input("拍照检测", disabled=not enable)

if img_file_buffer is not None:
    # 读取图像
    bytes_data = img_file_buffer.getvalue()
    
    # 使用 OpenCV 读取
    img_array = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # 显示原图和处理结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, channels="BGR", caption="原图", use_column_width=True)
    
    # YOLO检测
    with st.spinner("检测中..."):
        results = model(img, conf=conf_threshold, verbose=False)
        annotated_img = results[0].plot()  # 返回BGR格式
    
    with col2:
        st.image(annotated_img, channels="BGR", caption="检测结果", use_column_width=True)
    
    # 显示检测统计
    if results[0].boxes is not None:
        boxes = results[0].boxes
        count = len(boxes)
        
        if count > 0:
            st.success(f"✅ 检测到 {count} 个目标")
            
            # 统计各类别数量
            class_counts = {}
            for box in boxes:
                cls = int(box.cls[0].item())
                name = results[0].names[cls]
                class_counts[name] = class_counts.get(name, 0) + 1
            
            # 显示详细统计
            st.write("检测详情：")
            for name, cnt in class_counts.items():
                st.write(f"- {name}: {cnt}")
        else:
            st.info("未检测到目标")
