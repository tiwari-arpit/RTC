import streamlit as st
import requests
import tempfile

st.set_page_config(page_title="Vision Pipeline", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        height: 100%;
        overflow: hidden;
    }
    .block-container {
        padding: 1rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Real-Time Vision Pipeline")

left, right = st.columns([1, 3])

with left:
    st.subheader("Controls")

    source_type = st.radio("Video Source", ["Webcam", "Upload Video"])
    video_path = "0"

    if source_type == "Upload Video":
        uploaded = st.file_uploader("Upload Video", type=["mp4", "avi"])
        if uploaded:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded.read())
            video_path = temp_file.name

    caption = st.checkbox("Enable Captioning", value=False)
    save = st.checkbox("Save Output Video")

    save_path = "output.mp4" if save else ""

    col1, col2 = st.columns(2)

    if "running" not in st.session_state:
        st.session_state.running = False

    with col1:
        if st.button("Start", disabled=st.session_state.running):
            requests.get(
                "http://localhost:8000/start",
                params={
                    "source": video_path,
                    "no_caption": not caption,
                    "save": save_path
                }
            )
            st.session_state.running = True

    with col2:
        if st.button("Stop", disabled=not st.session_state.running):
            requests.get("http://localhost:8000/stop")
            st.session_state.running = False

    if st.session_state.running:
        requests.get(
            "http://localhost:8000/caption",
            params={"enable": caption}
        )

with right:
    st.subheader("Live Stream")

    st.markdown(
        """
        <div style="height:80vh; display:flex; align-items:center; justify-content:center;">
            <img src="http://localhost:8000/video" style="max-height:100%; max-width:100%;">
        </div>
        """,
        unsafe_allow_html=True
    )