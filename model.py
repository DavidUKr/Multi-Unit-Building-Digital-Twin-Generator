import numpy as np
from PIL import Image
import cv2
import utils
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import net.mtfsm as mtfsm

net=mtfsm.get_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_checkpoint=torch.load('models/sanity-room_loss/mtfsm_san_10s.pth', map_location=device)
net.load_state_dict(model_checkpoint)
net.eval()

def get_predictions(image_file):
    image = cv2.imread(image_file)
    width, height, _ = image_rgb.shape
    image_resized=cv2.resize(image, (1024,1024), interpolation=cv2.INTER_AREA)
    image_rgb=cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    predictions= run(image_rgb)

    prediction_pngs = utils.tensor_to_pngs(predictions["segmentation"], target_width=width, target_height=height)

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
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor =transform(image).unsqueeze(0)  # Shape: (1, 3, 1024, 1024)
    input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor

    with torch.no_grad():
            wall_pred, room_pred, graph, room_loss = net(input_tensor)

    wall_mask = torch.argmax(wall_pred, dim=1).squeeze(0)  # Shape: (1024, 1024)
    room_mask = torch.argmax(room_pred, dim=1).squeeze(0)
    # TODO: process graph
    
    combined_mask=room_mask.clone()
    #shift room class values
    non_zero_px=(combined_mask !=0)
    combined_mask[non_zero_px]+=3

    for i in range(1,4): # skip 0 - background
        wall_pixels= (wall_mask == i)
        combined_mask[wall_pixels]=i

    sep_masks=F.one_hot(combined_mask, num_classes=10)
    sep_masks=sep_masks.permute(2,0,1)


    output = {
    "segmentation": sep_masks,
    "graph": {
        "nodes": [
            {"id": 0, "x": 100, "y": 200, "type": "appartment_unit", "confidence": 0.95},
            {"id": 1, "x": 300, "y": 250, "type": "hallway", "confidence": 0.90},
            {"id": 2, "x": 400, "y": 150, "type": "stairwell", "confidence": 0.94},
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
        "doorway":2,
        "window": 3,
        "appartment_unit": 4,
        "hallway": 5,
        "elevator": 6,
        "stairwell": 7,
        "public_ammenity": 8,
        "balcony": 9
    },
}

    return output