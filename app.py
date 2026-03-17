import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import torch

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 口罩佩戴检测系统")
st.markdown("上传视频进行口罩佩戴检测")

# 模型加载（修复PyTorch 2.6兼容性问题）
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("请将 best.pt 文件放在项目根目录")
        st.stop()
    
    # 修复：使用weights_only=False加载模型
    try:
        # 方法1：直接加载（适用于最新版ultralytics）
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.info("尝试使用兼容模式加载...")
        
        # 方法2：使用torch.load先加载再传给YOLO
        try:
            # 设置weights_only=False以兼容旧模型
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            model = YOLO(model_path)
            return model
        except Exception as e2:
            st.error(f"兼容模式也失败: {e2}")
            st.stop()

# 加载模型
with st.spinner("正在加载模型..."):
    model = load_model()
st.success("✅ 模型加载成功")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 设置")
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25, 0.05)
    
    # 显示模型信息
    if model:
        st.markdown("---")
        st.markdown("### 模型信息")
        st.write(f"模型名称: {model.__class__.__name__}")
        if hasattr(model, 'names'):
            st.write(f"类别数: {len(model.names)}")
            st.write("类别:", list(model.names.values()))

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
        try:
            with st.spinner("正在检测中，请稍候..."):
                # 打开视频
                cap = cv2.VideoCapture(input_path)
                
                # 获取视频信息
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames <= 0:
                    st.error("无法读取视频文件")
                    st.stop()
                
                # 创建视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_count = 0
                
                # 读取并处理每一帧
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # YOLO检测
                    results = model(frame, conf=conf_threshold, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    # 写入处理后的帧
                    out.write(annotated_frame)
                    
                    # 更新进度
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"处理进度: {frame_count}/{total_frames} 帧")
                
                # 释放资源
                cap.release()
                out.release()
                
                status_text.text("处理完成!")
                st.success("✅ 检测完成！")
                
                # 提供下载
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                    st.download_button(
                        label="📥 下载检测结果",
                        data=video_bytes,
                        file_name=f"detected_{uploaded_file.name}",
                        mime="video/mp4",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"处理过程中出现错误: {e}")
        
        finally:
            # 清理临时文件
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass
