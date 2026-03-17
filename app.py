import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
from ultralytics import YOLO
import av
import os

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 实时口罩检测")
st.markdown("实时视频流检测")

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

# 视频处理类
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # YOLO检测
        results = model(img, conf=conf_threshold, verbose=False)
        annotated_img = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 启动WebRTC流
webrtc_streamer(
    key="face-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
