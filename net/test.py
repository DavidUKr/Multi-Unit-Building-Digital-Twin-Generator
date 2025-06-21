import mtfsm as model
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics import JaccardIndex
from scipy import ndimage
from acw_loss import ACW_loss
import os, re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import json

from dataset import FloorplanDataset


net = model.get_model()
net.eval()

test_epoch=480
test_image=20
models_folder=False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if models_folder:
    net_checkpoint = torch.load(f'../models/sanity_checks/mtfsm_sc_1s.pth', map_location=device)
    # net.load_state_dict(net_checkpoint)
    net.load_state_dict(net_checkpoint['model_state_dict'])
else:
    net_checkpoint = torch.load(f'checkpoints/trained_model_ep{test_epoch}.pth', map_location=device)
    net.load_state_dict(net_checkpoint['model_state_dict'])

net.eval()

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #for resnet pretrained
])

def find_png_by_index(search_dir, index= 14):
    
    number = str(index)
    pattern = rf'r_{number}_([0-9a-f]{{40}})\.png'
    
    if not os.path.exists(search_dir):
        print(f"Directory '{search_dir}' not found.")
        return None
    
    for filename in os.listdir(search_dir):
        match = re.match(pattern, filename)
        if match:
            filename = f"r_{number}_{match.group(1)}.png"
            path = os.path.join(search_dir, filename)
            
            # Check if original file exists in floorplans/
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    return path
                except Exception as e:
                    print(f"Error displaying image: {e}")
                    return path
            else:
                print(f"Original file {filename} not found in '{search_dir}'.")
                return None
    
    print(f"No resized PNG file found for number {number} in '{search_dir}'.")
    return None

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

def test():
    input_image_path=find_png_by_index('../train_data/mufp_10/val/JPEGImages', index=test_image)
    gt_path=find_png_by_index('../train_data/mufp_10/val/SegmentationClass', index=test_image)

    input_image = Image.open(input_image_path).convert('RGB')
    input_image = transform(input_image)
    input_image = input_image.unsqueeze(0)  # Shape: (1, 3, 1024, 1024)

    gt_image = Image.open(gt_path).convert('RGB')
    train_dataset= FloorplanDataset(dataset_dir="../train_data/mufp_10")
    gt_np= train_dataset.split_rgb_mask_per_class(gt_image)

    with torch.no_grad():
        wall_pred, room_pred, graph, room_loss = net(input_image)

    wall_mask = torch.argmax(wall_pred, dim=1).squeeze(0)  # Shape: (1024, 1024)
    room_mask = torch.argmax(room_pred, dim=1).squeeze(0)

    combined_mask=room_mask.clone()
    wall_filtered_mask=room_mask.clone()
    #shift room class values
    non_zero_px=(combined_mask !=0)
    combined_mask[non_zero_px]+=3
    

    for i in range(1,4): # without 0 - background
        wall_class_pixels= (wall_mask == i)
        combined_mask[wall_class_pixels]=i
        #filter out only walls
        if i==1:
            wall_filtered_mask[wall_class_pixels]=0

    sep_masks=F.one_hot(combined_mask, num_classes=10)
    sep_masks=sep_masks.permute(2,0,1)
    
    graph_np=graph.squeeze().cpu().numpy()
    wall_filt_np=wall_filtered_mask.cpu().numpy()

    room_graph, room_ids, room_pos, room_names=get_graph_from_prediction(
        wall_filt_np, 
        graph_np, 
        ['appartment_unit', 'hallway', 'elevator', 
        'stairwell', 'public_ammenity', 'balcony'])

    print("Values")
    print("wall", np.unique(wall_mask))
    print("room", np.unique(room_mask))
    print("combined", np.unique(combined_mask))

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes=axes.flatten()
    show_image = input_image.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()  # Shape: (1024, 1024, 3)
    show_image=show_image* np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    axes[0].imshow(show_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(wall_mask.cpu().numpy(), cmap='jet', vmin=0, vmax=3)  # Specify vmin/vmax for clarity
    axes[1].set_title('Structural Elements (Wall Mask)')
    axes[1].axis('off')

    axes[2].imshow(room_mask.cpu().numpy(), cmap='jet', vmin=0, vmax=6)  # Specify vmin/vmax for 7 classes
    axes[2].set_title('Functional Spaces (Room Mask)')
    axes[2].axis('off')

    axes[3].imshow(wall_filt_np, cmap='jet') 
    axes[3].set_title('Connnection Graph')
    axes[3].axis('off')

    G_rooms = nx.from_numpy_array(room_graph)
    relabel_mapping = {i: room_ids[i] for i in range(len(room_ids))}
    G_rooms = nx.relabel_nodes(G_rooms, relabel_mapping, copy=False)
    
    print("Graph edges min:", np.min([data['weight'] for u,v,data in G_rooms.edges(data=True)]))
    print("Graph edges max:", np.max([data['weight'] for u,v,data in G_rooms.edges(data=True)]))
    # Filter out weak edges based on the threshold
    conn_threshold=4
    edges_to_remove = [(u, v) for u, v, data in G_rooms.edges(data=True) if data['weight'] < conn_threshold]
    G_rooms.remove_edges_from(edges_to_remove)

    #test graph json
    G_safe = make_graph_json_safe(G_rooms, room_pos, room_names)
    graph_dict= json_graph.node_link_data(G_safe)
    graph_json= json.dumps (graph_dict, indent=4)
    with open("output/graph_response.json", "w") as f:
        f.write(graph_json)

    #Color the nodes:
    cmap = plt.get_cmap('tab10')
    color_map = {
        'appartment_unit': cmap(0),
        'hallway':         cmap(1),
        'elevator':        cmap(2),
        'stairwell':       cmap(3),
        'public_ammenity': cmap(4),
        'balcony':         cmap(5),
    }
    default_color = 'gray'

    node_color_list = []
    for node in G_rooms.nodes():
        class_name = room_names[node]
        node_color_list.append(color_map.get(class_name, default_color))

    # Draw the final graph
    nx.draw(
        G_rooms,
        ax=axes[3], # Suggest drawing on the room mask (axes[2]) for better context
        pos=room_pos,
        labels=room_names,
        with_labels=True,
        node_color=node_color_list,
        node_size=1500,    
        edgecolors='black',
        edge_color='black',
        width=2.0,         
        font_size=8,       
        font_weight='bold'
    )

    axes[4].imshow(combined_mask.cpu().numpy(), cmap='jet')  # Specify vmin/vmax for 7 classes
    axes[4].set_title('Combined Mask')
    axes[4].axis('off')

    axes[5].imshow(gt_np, cmap='jet') 
    axes[5].set_title('Ground Truth')
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('output/pred.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2,5, figsize=(20, 10))
    axes=axes.flatten()
    for i in range(10):
        axes[i].imshow(sep_masks[i], cmap='jet')
        axes[i].set_title(f'Mask #{i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

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
    # Create a new graph to avoid modifying the original
    H = nx.Graph()
    
    # Process nodes
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
                   original_id=str(node) # Keep track of the original tuple ID
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

if __name__ == '__main__':
    test()
