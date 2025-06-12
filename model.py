import numpy as np
from PIL import Image
import utils
import torch
import torchvision.transforms as transforms

import net.res_u_net as net

# model=net.get_model()
# model=model.cuda() if torch.cuda.is_available() else model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_checkpoint=torch.load('net/test_model.pth', map_location=device)
# model.load_state_dict(model_checkpoint)
# model.eval()
# # checkpoint = torch.load("path_to_checkpoint.pth")
# model.load_state_dict(checkpoint['model_state_dict'])

def get_predictions(image_file):
    image_rgb = Image.open(image_file).convert("RGB" )

    #save for dimension integrity
    width, height = image_rgb.size

    predictions= run(image_rgb)

    prediction_pngs = utils.tensor_to_pngs(predictions["segmentation"], target_width=width, target_height=height, threshold=0.9)

    output = {
        "segmentation": [
            {"class": class_name, "png": png}
            for class_name, png in zip(predictions["class_map"].keys(), prediction_pngs)
        ],
        "graph": predictions["graph"],
        "class_map": predictions["class_map"]
    }

    return output

def run(image):
    # preprocess = transforms.Compose([
    #         transforms.ToTensor(),  # Converts (H, W, C) to (C, H, W), scales [0, 255] to [0, 1]
    #         # transforms.Normalize(),
    #     ])
    # input_tensor = preprocess(image).unsqueeze(0)  # Shape: (1, 3, 1024, 1024)
    # input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor

    # with torch.no_grad():
    #         output = model(input_tensor)  # Shape: (1, num_classes, 1024, 1024)
    #         predictions = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Shape: (1024, 1024)

    #     # Convert predictions to a JSON-serializable format
    # wall_output = predictions.tolist()  # Convert NumPy array to list for JSON


    output = {
    "segmentation": torch.randn(512, 512, 9),  # 512x512x9 for 6 space + 3 object classes
    "graph": {
        "nodes": [
            {"id": 0, "x": 100, "y": 200, "type": "bathroom", "confidence": 0.95},
            {"id": 1, "x": 300, "y": 250, "type": "hallway", "confidence": 0.90},
            {"id": 2, "x": 400, "y": 150, "type": "hallway", "confidence": 0.94},
            # ... dynamic number of nodes
        ],
        "edges": [
            [0,1],
            [1,2]
            # ... dynamic number of edges
        ]
    },
    "class_map":{
        "background":0,
        "wall":1,
        "oppening":2,
        "bathroom": 3,
        "hallway": 4,
        "appartment_unit": 5,
        "elevator": 6,
        "stairwell": 7,
        "leisure_area": 8
    },
}

    return output