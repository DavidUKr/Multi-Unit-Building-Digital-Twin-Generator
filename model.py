import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from scipy import ndimage
from PIL import Image
import cv2
import utils
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import json

import net.mtfsm as mtfsm

net=mtfsm.get_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_checkpoint=torch.load('models/sanity-room_loss/mtfsm_san_10s.pth', map_location=device)
net.load_state_dict(model_checkpoint)
net.eval()

def get_predictions(image_file):

    try:
         file_bytes=image_file.read()
         np_array=np.frombuffer(file_bytes, np.uint8)
         image=cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED) #keep alpha channel
    except Exception as e:
         print(e)
         return "Could not read image file"

    height, width, _ = image.shape
    image=process_image(image)

    predictions= run(image)

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
    wall_filtered_mask=room_mask.clone()
    #shift room class values
    non_zero_px=(combined_mask !=0)
    combined_mask[non_zero_px]+=3

    for i in range(1,4): # skip 0 - background
        wall_class_pixels= (wall_mask == i)
        combined_mask[wall_class_pixels] = i
        #filter out just walls
        if i==1:
            wall_filtered_mask[wall_class_pixels]=0
    #separate masks per class
    sep_masks=F.one_hot(combined_mask, num_classes=10)
    sep_masks=sep_masks.permute(2,0,1)
    #graph calculation
    graph_np=graph.squeeze().cpu().numpy()
    wall_filt_np=wall_filtered_mask.cpu().numpy()

    room_graph, room_ids, room_pos, room_names=get_graph_from_prediction(
        wall_filt_np, 
        graph_np, 
        ['appartment_unit', 'hallway', 'elevator', 
        'stairwell', 'public_ammenity', 'balcony'])
    
    G_rooms = nx.from_numpy_array(room_graph)
    relabel_mapping = {i: room_ids[i] for i in range(len(room_ids))}
    G_rooms = nx.relabel_nodes(G_rooms, relabel_mapping, copy=False)
    
    # Filter out weak edges based on the threshold
    conn_threshold=4
    edges_to_remove = [(u, v) for u, v, data in G_rooms.edges(data=True) if data['weight'] < conn_threshold]
    G_rooms.remove_edges_from(edges_to_remove)    

    G_safe= make_graph_json_safe(G_rooms, room_pos, room_names)
    graph_dict= json_graph.node_link_data(G_safe)
    graph_json= json.dumps (graph_dict, indent=4)

    output = {
    "segmentation": sep_masks,
    "graph": graph_json,
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

def process_image(img, target_size=(1024,1024), enhance_brightness=True,
                 black_color=(0, 165, 255), white_color=(128, 0, 128), 
                 black_threshold=20, white_threshold=235, alpha_threshold=10):
     
    img_remapped = remap_black_white_pixels_rgba(img, black_color=black_color, white_color=white_color,
                                                black_threshold=black_threshold, white_threshold=white_threshold,
                                                alpha_threshold=alpha_threshold)
    
    img_bgr = convert_rgba_to_bgr(img_remapped, background_color=(0, 0, 0))

    resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    if enhance_brightness:
        resized = adjust_brightness_contrast(resized, alpha=2, beta=0)

    return resized
    

def adjust_brightness_contrast(image, alpha=1.1, beta=10):
    """
    Adjust brightness and contrast of an image.
    alpha: contrast control (1.0 = no change, >1.0 increases contrast)
    beta: brightness control (0 = no change, positive increases brightness)
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def remap_black_white_pixels_rgba(image, black_color=(0, 165, 255), white_color=(128, 0, 128), 
                                 black_threshold=20, white_threshold=235, alpha_threshold=10):
    """
    Remap black and white pixels in an RGBA image, only for non-transparent pixels.
    black_color: BGR color to replace black pixels (default: bright orange)
    white_color: BGR color to replace white pixels (default: purple)
    black_threshold: Pixels with all RGB channels <= this value are considered black
    white_threshold: Pixels with all RGB channels >= this value are considered white
    alpha_threshold: Pixels with alpha <= this value are considered transparent
    """
    if image.shape[2] != 4:
        return image  # If not RGBA, return unchanged
    
    # Split into RGB and alpha
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]
    
    # Create masks for non-transparent black and white pixels
    non_transparent_mask = alpha > alpha_threshold
    black_mask = np.all(rgb <= black_threshold, axis=2) & non_transparent_mask
    white_mask = np.all(rgb >= white_threshold, axis=2) & non_transparent_mask
    
    # Create output image as a copy
    output = image.copy()
    
    # Replace black pixels with bright orange (in BGR)
    output[black_mask, :3] = black_color
    
    # Replace white pixels with purple (in BGR)
    output[white_mask, :3] = white_color
    
    return output

def convert_rgba_to_bgr(image, background_color=(0, 0, 0)):
    """
    Convert RGBA image to BGR, blending transparent areas with background_color.
    """
    if image.shape[2] == 4:  # RGBA image
        rgb = image[:, :, :3]
        alpha = image[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
        background = np.array(background_color, dtype=np.uint8)
        bgr = (alpha[:, :, np.newaxis] * rgb + (1 - alpha[:, :, np.newaxis]) * background).astype(np.uint8)
        return bgr
    return image  # Already BGR

def get_graph_from_prediction(mask, A, classes, image_dim=1024, graph_dim=32):
    classmap={idx+1:cls for idx, cls in enumerate(classes)}
    #calculate patches (node coverage) and assign pixels to node
    patch_size = image_dim // graph_dim
    
    y_patches = np.arange(image_dim) // patch_size
    x_patches = np.arange(image_dim) // patch_size

    pixel_to_graph_node_map = y_patches[:, np.newaxis] * graph_dim + x_patches[np.newaxis, :]
    
    instance_pos={}
    instance_labels={}
    instance_to_graph_nodes= {}

    #idenitify room instances
    for room_id, name in classmap.items():
        class_mask= (mask == room_id)
        #morphological opening to remove small, noisy islands of pixels.
        class_mask = ndimage.binary_opening(class_mask, iterations=2)

        labeled_array, num_instances=ndimage.label(class_mask)

        if num_instances > 0:
            print(f"Found {num_instances} instance(s) of '{name}' (ID: {room_id})")
        
        for i in range(1, num_instances+1):
            instance_id= (room_id, i)
            instance_mask = (labeled_array == i)

            y_coords, x_coords = np.where(instance_mask)

            MINIMUM_PIXEL_AREA = 50 # Example value, tune this
            if len(y_coords) < MINIMUM_PIXEL_AREA:
                continue

            y_mean= int(y_coords.mean())
            x_mean= int(x_coords.mean())

            instance_pos[instance_id]=(x_mean, y_mean)
            instance_labels[instance_id]=name
            #assign instance to one or mulitple graph nodes (a node can be assigned to multiple instances)
            instance_to_graph_nodes[instance_id] = np.unique(pixel_to_graph_node_map[instance_mask])

    instance_ids=list(instance_pos.keys())
    num_instances=len(instance_ids)
    instance_adj_matrix=np.zeros((num_instances, num_instances))

    #build adjacency matrix between instances by calculating strength between covered nodes
    for i in range(num_instances):
        for j in range(i, num_instances):
            inst1_id=instance_ids[i]
            inst2_id=instance_ids[j]
            
            i1_graph_nodes=instance_to_graph_nodes[inst1_id]
            i2_graph_nodes=instance_to_graph_nodes[inst2_id]

            if len(i1_graph_nodes)==0 or len(i2_graph_nodes)==0:
                continue

            sub_matrix = A[np.ix_(i1_graph_nodes, i2_graph_nodes)]
            connection_strength = sub_matrix.sum()

            instance_adj_matrix[i,j]=connection_strength
            instance_adj_matrix[j,i]=connection_strength

    return instance_adj_matrix, instance_ids, instance_pos, instance_labels

def make_graph_json_safe(G, node_positions, node_labels):
    """
    Prepares a NetworkX graph for JSON serialization by converting all attributes
    and node IDs into JSON-compatible data types.
    
    Args:
        G (nx.Graph): The graph to process.
        node_positions (dict): Dictionary mapping node IDs to (x, y) positions.
        node_labels (dict): Dictionary mapping node IDs to string labels.
        
    Returns:
        nx.Graph: A new graph with all attributes converted to JSON-safe types.
    """
    H = nx.Graph()
    
    for node, data in G.nodes(data=True):
        # Convert the tuple node ID to a string for universal compatibility
        node_id_str = str(node)
        
        # Get attributes and convert types
        label = node_labels.get(node, "")
        pos = node_positions.get(node, (0, 0))
        
        H.add_node(node_id_str,
                   label=str(label),
                   x=int(pos[0]),  # Convert numpy.int64 to native Python int
                   y=int(pos[1]),
                  )
        
    # Process edges
    for u, v, data in G.edges(data=True):
        u_str = str(u)
        v_str = str(v)
        
        weight = data.get('weight', 0.0)
        
        H.add_edge(u_str, v_str,
                   weight=float(weight) # Convert numpy.float64 to native Python float
                  )
        
    return H