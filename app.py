import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from PIL import Image
import os

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 口罩佩戴检测系统")
st.markdown("上传视频进行口罩佩戴检测")

# 模型加载
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
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("1. 上传视频文件")
    st.markdown("2. 等待检测完成")
    st.markdown("3. 下载检测后的视频")

# 主界面
uploaded_file = st.file_uploader("选择视频文件", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # 保存上传的视频
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
        tmp_input.write(uploaded_file.getvalue())
        input_path = tmp_input.name
    
    # 创建输出视频路径
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    # 显示原视频
    st.video(uploaded_file)
    
    # 检测按钮
    if st.button("开始检测", type="primary"):
        with st.spinner("正在检测中，请稍候..."):
            # 打开视频
            cap = cv2.VideoCapture(input_path)
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 创建进度条
            progress_bar = st.progress(0)
            frame_count = 0
            
            # 读取并处理每一帧
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # YOLO检测
                results = model(frame, conf=conf_threshold)
                annotated_frame = results[0].plot()
                
                # 写入处理后的帧
                out.write(annotated_frame)
                
                # 更新进度
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
            
            # 释放资源
            cap.release()
            out.release()
            
            st.success("✅ 检测完成！")
            
            # 提供下载
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
                st.download_button(
                    label="下载检测结果",
                    data=video_bytes,
                    file_name="detected_" + uploaded_file.name,
                    mime="video/mp4"
                )
            
            # 清理临时文件
            os.unlink(input_path)
            os.unlink(output_path)
