import collections

class TemporalBuffer:
    def __init__(self, window_size):
        self.window_size = window_size
        self.tracks = collections.defaultdict(lambda: collections.deque(maxlen=window_size))
        self.frame_buffer = collections.deque(maxlen=window_size)
        self.frame_idx = 0

    def update(self, tracked_objects, frame):
        self.frame_buffer.append(frame.copy())
        for obj in tracked_objects:
            self.tracks[obj["track_id"]].append({
                "box": obj["box"],
                "cls_name": obj["cls_name"],
                "conf": obj["conf"],
                "frame": self.frame_idx
            })
        self.frame_idx += 1

    def get_scene_summary(self):
        class_counts = collections.Counter()
        active_tracks = []

        for tid, entries in self.tracks.items():
            if entries:
                recent = entries[-1]
                if self.frame_idx - recent["frame"] <= self.window_size:
                    class_counts[recent["cls_name"]] += 1
                    active_tracks.append(tid)

        rep_frame = None
        if self.frame_buffer:
            mid = len(self.frame_buffer)//2
            rep_frame = list(self.frame_buffer)[mid]

        return {
            "class_counts": dict(class_counts),
            "active_tracks": active_tracks,
            "representative_frame": rep_frame
        }