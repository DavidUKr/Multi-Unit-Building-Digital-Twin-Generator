import mtfsm as model
import torch
from torchvision import transforms
import os, re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



net = model.get_model()
net.eval()

test_epoch=400
test_image=22

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net_checkpoint = torch.load(f'checkpoints/trained_model_ep{test_epoch}.pth', map_location=device)
# net_checkpoint = torch.load(f'../models/test_model.pth', map_location=device)
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

def test():
    input_image_path=find_png_by_index('../train_data/mufp_10/val/JPEGImages', index=test_image)
    gt_path=find_png_by_index('../train_data/mufp_10/val/SegmentationClass', index=test_image)

    input_image = Image.open(input_image_path).convert('RGB')
    input_image = transform(input_image)
    input_image = input_image.unsqueeze(0)  # Shape: (1, 3, 1024, 1024)

    with torch.no_grad():
        wall_pred, room_pred, graph, room_loss = net(input_image)

    print("wall unique values before argmax", np.unique(wall_pred))
    print("room", np.unique(room_pred))

    wall_mask = torch.argmax(wall_pred, dim=1).squeeze(0)  # Shape: (1024, 1024)
    room_mask = torch.argmax(room_pred, dim=1).squeeze(0)  

    print("wall after argmax", np.unique(wall_mask))
    print("room", np.unique(room_mask))

    #ground truth
    gt_mask= Image.open(gt_path).convert('RGB')
    gt_mask_np=np.array(gt_mask)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

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

    # axes[3].imshow(Image.open(gt_path), cmap='jet')
    # axes[3].set_title('Ground Truth')
    # axes[3].axis('off')

    
    plt.tight_layout()
    plt.savefig('output/pred.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    test()
