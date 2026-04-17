import cv2
from config import PALETTE


class OutputRenderer:
    """Renders bounding boxes, track IDs, and captions onto the frame."""

    @staticmethod
    def draw_tracks(frame, tracked_objects):
        for obj in tracked_objects:
            tid = obj["track_id"]
            x1, y1, x2, y2 = obj["box"]
            color = PALETTE[tid % len(PALETTE)]

            label = f"#{tid} {obj['cls_name']} {obj['conf']:.2f}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1
            )

            cv2.rectangle(
                frame,
                (x1, y1 - th - 8),
                (x1 + tw + 6, y1),
                color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return frame

    @staticmethod
    def draw_captions(frame, captions):
        h, w = frame.shape[:2]
        y_start = h - 20 - (len(captions) * 24)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, y_start - 10),
            (w, h),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        for i, cap in enumerate(captions):
            y = y_start + i * 24
            cv2.putText(
                frame,
                cap,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 200),
                1,
                cv2.LINE_AA
            )

        return frame

    @staticmethod
    def draw_hud(frame, fps, n_tracks, device):
        info = f"FPS:{fps:.1f}  Tracks:{n_tracks}  [{device.upper()}]"

        cv2.putText(
            frame,
            info,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        return frame