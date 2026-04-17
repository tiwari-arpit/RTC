import time, collections, cv2
from config import *
from core.video_stream import VideoStream
from core.preprocessor import FramePreprocessor
from core.tracker import MultiObjectTracker
from core.buffer import TemporalBuffer
from core.captioner import CaptionGenerator
from core.renderer import OutputRenderer

class RealtimePipeline:

    def __init__(self, video_source, use_caption=True, save_output=None):
        self.stream = VideoStream(video_source)
        self.preprocessor = FramePreprocessor()
        self.tracker = MultiObjectTracker(YOLO_MODEL, CONF_THRESHOLD, IOU_THRESHOLD)
        self.buffer = TemporalBuffer(TEMPORAL_WINDOW)
        self.renderer = OutputRenderer()
        self.use_caption = use_caption

        self.captioner = CaptionGenerator() if use_caption else None

        self.caption_history = collections.deque(maxlen=MAX_DISPLAY_CAPTIONS)
        self.frame_count = 0
        self.fps_timer = time.time()
        self.fps_display = 0.0

        self.writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                save_output, fourcc, 30.0,
                (self.stream.width, self.stream.height)
            )

    def run(self):
        while True:
            frame = self.stream.read()
            if frame is None:
                break

            self.frame_count += 1

            tracked = self.tracker.track(frame)
            self.buffer.update(tracked, frame)

            if self.use_caption and self.frame_count % CAPTION_INTERVAL == 0:
                scene = self.buffer.get_scene_summary()
                rep = scene["representative_frame"]
                if rep is not None:
                    cap = self.captioner.generate_scene_caption(rep, scene)
                    self.caption_history.append(cap)

            display = self.renderer.draw_tracks(frame.copy(), tracked)
            display = self.renderer.draw_captions(display, list(self.caption_history))
            display = self.renderer.draw_hud(display, self.fps_display, len(tracked), DEVICE)

            cv2.imshow("Pipeline", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stream.release()
        cv2.destroyAllWindows()