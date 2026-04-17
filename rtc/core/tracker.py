from ultralytics import YOLO

class MultiObjectTracker:
    def __init__(self, model_name, conf, iou):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou

    def track(self, frame):
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )[0]

        tracked = []
        if results.boxes is None or results.boxes.id is None:
            return tracked

        for box, tid in zip(results.boxes, results.boxes.id):
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            tracked.append({
                "track_id": int(tid),
                "box": (x1,y1,x2,y2),
                "conf": float(box.conf[0]),
                "cls_id": int(box.cls[0]),
                "cls_name": results.names[int(box.cls[0])]
            })
        return tracked