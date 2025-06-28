from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
from io import BytesIO

# Load SAM model
checkpoint = "./models/sam_vit_h_4b8939.pth"  # Update with your checkpoint path
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
predictor = SamPredictor(sam)

def get_embedding(image_file):
    image_data = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)
    embedding = predictor.get_image_embedding().cpu().numpy()  # Shape: [1, 256, 64, 64]

    return embedding.astype(np.float32).tobytes()