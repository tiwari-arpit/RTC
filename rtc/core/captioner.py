from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import cv2
from PIL import Image
from config import DEVICE


class CaptionGenerator:
    def __init__(self):
        print("[CAPTION] Using BLIP-base (FAST)")
        self.device = DEVICE

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

        if self.device == "cuda":
            self.model = self.model.half()

        self.model.eval()

    @torch.no_grad()
    def generate_scene_caption(self, frame, scene_summary):
        if frame is None:
            return "No scene"

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs = {k: v.half() for k, v in inputs.items()}

        out = self.model.generate(**inputs, max_new_tokens=30)

        return self.processor.decode(out[0], skip_special_tokens=True)