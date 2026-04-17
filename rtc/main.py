import cv2
import threading
import time
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

from core.pipeline import RealtimePipeline

app = FastAPI()

latest_frame = None
lock = threading.Lock()
pipeline_thread = None
stop_flag = False
caption_enabled = False


def run_pipeline(source=0, save_output=None):
    global latest_frame, stop_flag, caption_enabled

    stop_flag = False

    pipeline = RealtimePipeline(
        video_source=source,
        use_caption=True,
        save_output=save_output if save_output else None
    )

    while not stop_flag:
        frame = pipeline.stream.read()
        if frame is None:
            break

        pipeline.frame_count += 1

        tracked = pipeline.tracker.track(frame)
        pipeline.buffer.update(tracked, frame)

        display = pipeline.renderer.draw_tracks(frame.copy(), tracked)

        if caption_enabled and pipeline.frame_count % 60 == 0:
            scene = pipeline.buffer.get_scene_summary()
            rep = scene["representative_frame"]
            if rep is not None:
                try:
                    cap = pipeline.captioner.generate_scene_caption(rep, scene)
                    pipeline.caption_history.append(cap)
                except Exception:
                    pass

        display = pipeline.renderer.draw_captions(display, list(pipeline.caption_history))
        display = pipeline.renderer.draw_hud(display, 0.0, len(tracked), "CPU")

        with lock:
            latest_frame = display.copy()

    pipeline.stream.release()


def generate_frames():
    global latest_frame

    last = None

    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        with lock:
            frame = latest_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        if frame_bytes != last:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            last = frame_bytes

        time.sleep(0.04)


@app.get("/start")
def start_pipeline(source: str = "0", no_caption: bool = False, save: str = ""):
    global pipeline_thread, caption_enabled

    if pipeline_thread and pipeline_thread.is_alive():
        return JSONResponse({"status": "already running"})

    if not torch.cuda.is_available():
        no_caption = True

    caption_enabled = not no_caption

    src = int(source) if source.isdigit() else source

    pipeline_thread = threading.Thread(
        target=run_pipeline,
        args=(src, save if save else None),
        daemon=True
    )
    pipeline_thread.start()

    return {"status": "started"}


@app.get("/stop")
def stop_pipeline():
    global stop_flag
    stop_flag = True
    return {"status": "stopped"}


@app.get("/caption")
def toggle_caption(enable: bool):
    global caption_enabled
    caption_enabled = enable
    return {"caption": caption_enabled}


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/")
def root():
    return {"message": "API running"}