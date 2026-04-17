from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name, conf, iou):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append({
                "box":(x1,y1,x2,y2),
                "conf":float(box.conf[0]),
                "cls_id":int(box.cls[0]),
                "cls_name":results.names[int(box.cls[0])]
            })
        return detections