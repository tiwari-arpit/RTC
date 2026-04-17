import torch

VIDEO_SOURCE        = 0
YOLO_MODEL          = "yolov8n.pt"
CONF_THRESHOLD      = 0.4
IOU_THRESHOLD       = 0.5
TEMPORAL_WINDOW     = 8
CAPTION_INTERVAL    = 30
MAX_DISPLAY_CAPTIONS = 3
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
BLIP2_MODEL         = "Salesforce/blip2-opt-2.7b"

PALETTE = [
    (255,56,56),(255,157,51),(255,255,51),(51,255,255),
    (51,153,255),(153,51,255),(255,51,153),(51,255,153),
    (255,128,0),(0,255,128),(128,0,255),(0,128,255),
    (255,0,128),(128,255,0),(0,255,51),(51,0,255),
    (255,51,0),(0,51,255),(51,255,0),(0,255,204),
]