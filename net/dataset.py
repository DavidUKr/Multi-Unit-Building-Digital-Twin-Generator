from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class FloorplanDataset(Dataset):
    def __init__(self, dataset_dir, split='train', transform=None):
        
        self.transform=transform

        assert split in ['train', 'test', 'val'], "Split must be test, train or val"
        self.split=split
        self.data_dir=os.path.join(dataset_dir, split)
        self.image_paths= sorted(glob.glob(os.path.join(self.data_dir, "JPEGImages", "*.png")))
        self.mask_paths= sorted(glob.glob(os.path.join(self.data_dir, "SegmentationClass", "*.png")))
        #validation checks
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch in number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)})"
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            img_name = os.path.basename(img_path).split('.')[0]
            mask_name = os.path.basename(mask_path).split('.')[0]
            assert img_name == mask_name, f"Mismatch in filenames: {img_name}, {mask_name}"
        #rgb to class to index converison
        self.label_to_rgb=self._get_labelmap(os.path.join(dataset_dir, "labelmap.txt"))

        self.wall_classes = ['background', 'wall', 'doorway', 'window']
        self.room_classes = ['background','appartment_unit', 'hallway', 'elevator', 'stairwell', 'public_ammenity', 'balcony']

        self.wall_class_to_idx={cls:idx for idx, cls in enumerate(self.wall_classes)}
        self.room_class_to_idx={cls:idx for idx, cls in enumerate(self.room_classes)}

    def _get_labelmap(self, path):
        """
        Parse the labelmap.txt file to map labels to RGB colors.
        
        Args:
            labelmap_path (str): Path to labelmap.txt.
            
        Returns:
            dict: Mapping of label to RGB color string (e.g., '108,66,188').
        """
        labelmap={}
        with open (path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split(':')
                labelmap[parts[0]]=tuple(map(int, parts[1].split(','))) #labelmap[class]=color -> (int,int,int)

        return labelmap

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image= Image.open(self.image_paths[idx]).convert('RGB')
        mask= Image.open(self.mask_paths[idx]).convert('RGB')

        mask_np=np.array(mask)

        # unique_rgbs = np.unique(mask_np.reshape(-1, 3), axis=0)
        # print(f"Unique RGBs in mask {self.mask_paths[idx]}: {unique_rgbs}")

        wall_mask_np= np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
        room_mask_np= np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)

        for label, rgb in self.label_to_rgb.items():
            mask_pixels = (mask_np == rgb).all(axis=2)
            
            if label in self.wall_classes:
                wall_mask_np[mask_pixels]=self.wall_class_to_idx[label]
                room_mask_np[mask_pixels]=0
            elif label in self.room_classes:
                room_mask_np[mask_pixels]=self.room_class_to_idx[label]
                wall_mask_np[mask_pixels]=0
            else:
                NameError("Missmatched classes from dataset")

        if self.transform:
            image = self.transform(image)
            mask_transform = transforms.Compose([
                # transforms.ToPILImage(),  # Convert NumPy array to PIL Image for spatial transforms
                # # filter transforms from self.transforms for mask specific tranformations
                # *(t for t in self.transform.transforms 
                # if isinstance(t, (transforms.Resize, transforms.RandomCrop, 
                #                transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip))),
                transforms.ToTensor() # Converts [H, W] to [1, H, W]
            ])

            wall_mask = mask_transform(wall_mask_np).long().squeeze(0) # [1, H, W] -> [H, W]
            room_mask = mask_transform(room_mask_np).long().squeeze(0) # [1, H, W] -> [H, W]
        else:
            # Convert to tensors without transformation
            image = transforms.ToTensor()(image)  # [C, H, W]
            wall_mask = torch.tensor(wall_mask_np, dtype=torch.long)  # [H, W]
            room_mask = torch.tensor(room_mask_np, dtype=torch.long)  # [H, W]

        return image, wall_mask, room_mask
    

if __name__ == "__main__":

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #for resnet pretrained
    ])  

    train_dataset= FloorplanDataset(dataset_dir="data/mufp_10", split='train', transform=transform)
    # train_dataset= FloorplanDataset(dataset_dir="data/mufp_10", split='train')

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)

    count=0
    for i, (images, wall_masks, room_masks) in enumerate(train_loader):
        
        images = images.permute(0,2,3,1).numpy()
        wall_masks = wall_masks.numpy()
        room_masks = room_masks.numpy()

        print("room classes", np.unique(room_masks))
        print("wall classes", np.unique(wall_masks))

        for j in range(images.shape[0]):

            fig, axes = plt.subplots(1, 3, figsize=(15,5))

            img = images[j] * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            axes[0].imshow(img)
            axes[0].set_title(f'Sample {i*train_loader.batch_size+j}')
            axes[0].axis('off')

            axes[1].imshow(wall_masks[j], cmap='jet')  # Select channel for mask
            axes[1].set_title(f'Wall Mask {i*train_loader.batch_size+j}')
            axes[1].axis('off')

            axes[2].imshow(room_masks[j], cmap='jet')  # Select channel for mask
            axes[2].set_title(f'Room Mask {i*train_loader.batch_size+j}')
            axes[2].axis('off')

            # plt.show()

            # if j>=2:
            #     break
        
        count=count+1

    print(count)