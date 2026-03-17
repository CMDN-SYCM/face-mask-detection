import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
from ultralytics import YOLO
import os

st.set_page_config(page_title="口罩检测", layout="wide")

st.title("😷 跨设备实时口罩检测")
st.markdown("任何设备打开此页面，都使用自己的摄像头")

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
    
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("1. 点击 START 按钮")
    st.markdown("2. 允许摄像头权限")
    st.markdown("3. 实时检测开始")
    st.markdown("4. 手机/平板/电脑都可以用")

# 视频处理类
class VideoProcessor:
    def __init__(self):
        self.model = model
        self.conf_threshold = conf_threshold
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # YOLO检测
        results = self.model(img, conf=self.conf_threshold, verbose=False)
        annotated_img = results[0].plot()
        
        # 返回处理后的帧
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 配置STUN服务器（用于跨网络连接）
rtc_config = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# 启动WebRTC流
ctx = webrtc_streamer(
    key="face-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 显示状态
if ctx.state.playing:
    st.success("✅ 实时检测中...")
else:
    st.info("⏸️ 点击 START 开始检测")
