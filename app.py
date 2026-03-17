import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile

# ---------- 页面配置 ----------
st.set_page_config(page_title="YOLO实时检测", layout="wide")

# ---------- 配置STUN服务器（必须，用于部署）----------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------- 加载模型 ----------
@st.cache_resource
def load_model(model_path):
    """加载YOLO模型"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

def main():
    st.title("🎯 实时YOLO目标检测")
    st.markdown("---")
    
    # ---------- 模型文件处理 ----------
    # 方案1：如果你的pt文件已经在项目目录中
    model_path = "best.pt"  # 确保这个文件在你的项目根目录
    
    # 方案2：如果文件不在目录中，提供上传功能
    if not os.path.exists(model_path):
        st.warning("未在项目目录找到 best.pt 文件，请上传你的模型文件")
        uploaded_file = st.file_uploader("上传你的 best.pt 文件", type=['pt'])
        
        if uploaded_file is not None:
            # 保存上传的文件到临时目录
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                model_path = tmp_file.name
                st.success("模型上传成功！")
        else:
            st.info("请上传你的YOLO模型文件以继续")
            st.stop()
    
    # 加载模型
    with st.spinner("正在加载YOLO模型..."):
        model = load_model(model_path)
    
    if model is None:
        st.error("模型加载失败，请检查文件格式")
        st.stop()
    
    st.success(f"✅ 模型加载成功！")
    
    # ---------- 侧边栏配置 ----------
    with st.sidebar:
        st.header("⚙️ 检测配置")
        conf_threshold = st.slider(
            "置信度阈值", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.25, 
            step=0.05,
            help="只显示置信度高于此阈值的检测结果"
        )
        
        iou_threshold = st.slider(
            "IOU阈值", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.45, 
            step=0.05,
            help="非极大值抑制的IOU阈值"
        )
        
        img_size = st.selectbox(
            "输入图像尺寸",
            options=[320, 416, 512, 640],
            index=3,
            help="更大的尺寸精度更高但速度更慢"
        )
        
        st.markdown("---")
        st.header("📊 检测统计")
        detect_count = st.empty()  # 用于显示实时检测数量
        
        st.markdown("---")
        st.header("ℹ️ 使用说明")
        st.markdown("""
        1. 允许浏览器访问摄像头
        2. 点击"开始"按钮启动检测
        3. 可以调整阈值来过滤结果
        4. 点击"停止"结束检测
        """)
    
    # ---------- 视频处理类 ----------
    class YOLOVideoProcessor:
        def __init__(self):
            self.model = model
            self.conf_threshold = conf_threshold
            self.iou_threshold = iou_threshold
            self.img_size = img_size
            self.count = 0
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # 将帧转换为numpy数组
            img = frame.to_ndarray(format="bgr24")
            
            # 运行YOLO检测
            results = self.model.predict(
                img, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False  # 关闭详细输出
            )
            
            # 获取检测到的目标数量
            if len(results) > 0 and results[0].boxes is not None:
                self.count = len(results[0].boxes)
            
            # 绘制检测结果
            annotated_img = results[0].plot()
            
            # 返回处理后的帧
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
    
    # ---------- 启动WebRTC流 ----------
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🎥 实时检测画面")
        
        # 创建处理器实例
        processor = YOLOVideoProcessor()
        
        # 启动webrtc流
        ctx = webrtc_streamer(
            key="yolo-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: processor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # 如果正在检测，更新统计信息
        if ctx.state.playing:
            with st.sidebar:
                detect_count.metric("当前检测目标数", processor.count)
    
    # ---------- 底部说明 ----------
    st.markdown("---")
    st.markdown("""
    **注意事项：**
    - 首次使用需要允许摄像头权限
    - 如果画面卡顿，可以尝试降低图像尺寸
    - 部署到 Streamlit Cloud 时需要确保网络畅通
    """)

if __name__ == "__main__":
    main()
