import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchmetrics import JaccardIndex

import mtfsm as model
from acw_loss import ACW_loss
from dataset import FloorplanDataset

from memory_profiler import profile

#Hyperparameters
alpha=0.5
# lr=5e-4
lr=5e-4
weight_decay=2e-5
dropout=0.2
batch_size = 5
num_epochs = 500

# torch.cuda.empty_cache()

def validate_model(net, val_loader, wall_criterion, room_criterion, device, D=0.5, accuracy=False):
    net.eval()
    val_loss = 0.0
    wall_iou_metric = JaccardIndex(task='multiclass', num_classes=4).to(device)
    room_iou_metric = JaccardIndex(task='multiclass', num_classes=7).to(device)
    wall_correct, wall_total = 0, 0
    room_correct, room_total = 0, 0

    with torch.no_grad():
        for images, wall_masks, room_masks in val_loader:
            images, wall_masks, room_masks = images.to(device), wall_masks.to(device), room_masks.to(device)
            wall_out, room_out, graph, room_loss = net(images)
            wall_loss = wall_criterion(wall_out, wall_masks)
            room_acw_loss = room_criterion(room_out, room_masks)
            room_net_loss = room_loss + room_acw_loss
            total_loss = D * room_net_loss + (1 - D) * wall_loss
            val_loss += total_loss.item()

            # Compute IoU
            wall_preds = wall_out.argmax(dim=1)
            room_preds = room_out.argmax(dim=1)
            wall_iou_metric.update(wall_preds, wall_masks)
            room_iou_metric.update(room_preds, room_masks)

            # Compute pixel accuracy
            if accuracy:
                wall_correct += (wall_preds == wall_masks).sum().item()
                wall_total += wall_masks.numel()
                room_correct += (room_preds == room_masks).sum().item()
                room_total += room_masks.numel()
    net.train()

    avg_val_loss = val_loss / len(val_loader)
    wall_iou = wall_iou_metric.compute().item()
    room_iou = room_iou_metric.compute().item()
    if accuracy:
        wall_accuracy = wall_correct / wall_total
        room_accuracy = room_correct / room_total

        return avg_val_loss, wall_iou, room_iou, wall_accuracy, room_accuracy
    
    return avg_val_loss, wall_iou, room_iou

def check_val_perf(net, test_loader, wall_criterion, room_criterion, device, summary_writer=None, global_step=None, verbose=True):
    print("Validation performance")
    # avg_val_loss, wall_iou, room_iou, wall_accuracy, room_accuracy=validate_model(net, test_loader, wall_criterion, room_criterion, device)
    avg_val_loss, wall_iou, room_iou=validate_model(net, test_loader, wall_criterion, room_criterion, device)

    if summary_writer and global_step:
        summary_writer.add_scalar('avg_val_loss', avg_val_loss, global_step)
        summary_writer.add_scalar("wall_iou:", wall_iou, global_step)
        summary_writer.add_scalar("room_iou:", room_iou, global_step)
        # summary_writer.add_scalar("wall_accuracy:", wall_accuracy, global_step)
        # summary_writer.add_scalar("room_accuracy:", room_accuracy, global_step)

    if verbose:
        print("avg_val_loss:", avg_val_loss)
        print("wall_iou:", wall_iou)
        print("room_iou:", room_iou)
        # print("wall_accuracy:", wall_accuracy)
        # print("room_accuracy:", room_accuracy)

#Train loop
@profile
def train(resume_from_epoch=None):
    if torch.cuda.is_available():
        device='cuda'
    elif torch.backends.mps.is_available():
        device='mps'
    else:
        device='cpu'
    print("Loading model on ", device)
    
    net = model.get_model(pretrained_encoder=True, dropout=dropout).to(device)
    optimizer=optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = 0
    global_step = 0
    
    if resume_from_epoch:
        checkpoint_path = f'checkpoints/trained_model_ep{resume_from_epoch}.pth'
        print(f'Loading checkpoint: {checkpoint_path}')
        # Use a try-except block for safety
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            global_step = checkpoint.get('global_step', 0) # Use .get for backward compatibility
            print(f"Resumed from epoch {start_epoch}. Global step is {global_step}.")
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_path}. Starting from scratch.")

    print('Loading processes to device')
    
    wall_criterion=ACW_loss().to(device)
    room_criterion=ACW_loss().to(device)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #for resnet pretrained
    ])
    print("Loading dataset")
    train_dataset= FloorplanDataset(dataset_dir="../train_data/mufp_10", split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataset= FloorplanDataset(dataset_dir="../train_data/mufp_10", split='test', transform=transform)
    test_loader= DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Start training")
    writer = SummaryWriter("runs/mtfsm_san_run7-10s_wd")
    # writer.add_graph(net, train_dataset.__getitem__(0)[0].unsqueeze(0))
    writer.add_scalar('alpha', alpha, 0)
    writer.add_scalar('lr', lr, 0)
    writer.add_scalar('batch_size', batch_size, 0)
    writer.add_scalar('weight_decay', weight_decay, 0)
    writer.add_scalar('num_epochs', num_epochs, 0)
    

    net.train()

    for epoch in range(start_epoch, num_epochs):

        running_loss=0.0
        last_loss=0.0
        epoch_step=0

        for i, (images, wall_masks, room_masks) in enumerate(train_loader):
            images=images.to(device)
            wall_masks=wall_masks.to(device)
            room_masks=room_masks.to(device)
            #fw pass
            wall_out, room_out, graph_out, room_scg_loss=net(images)
            #loss
            wall_loss=wall_criterion(wall_out, wall_masks)
            room_loss=room_scg_loss+room_criterion(room_out, room_masks)
            # room_loss=room_criterion(room_out, room_masks)
            loss=alpha*room_loss + (1-alpha)*wall_loss
            
            #backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) #gradient clipping to prevent exploding gradients
            optimizer.step()

            loss_item=loss.item()

            print(f'Step {i} (global {global_step}) losses: wall: {wall_loss.item():.4f} room: {room_loss.item():.4f} total:  {loss_item:.4f}')

            #Visualization
            writer.add_scalar('Loss', loss_item, global_step)
            global_step += 1
            epoch_step += 1
            running_loss+= loss_item
            last_loss= loss_item
        
        print(f'Epoch [{epoch}/{num_epochs}] Loss: {running_loss/epoch_step:.4f} Last_Loss: {last_loss}')
        writer.add_scalar('running_loss', running_loss, global_step)
        
        if (epoch) % 10 == 0: #Validation at 10th epoch
            try:
                checkpoint_path = f'checkpoints/trained_model_ep{epoch}.pth'
                print(f"Saving checkpoint to {checkpoint_path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'loss': last_loss, # Or the average loss for the epoch
                }, checkpoint_path)
            except Exception as e:
                print("Could not save checkpoint -> error: ", e)
            check_val_perf(net, test_loader, wall_criterion, room_criterion, device, writer, global_step)

    #Saving and validating last epoch
    try:
        checkpoint_path = f'checkpoints/trained_model_ep{num_epochs}.pth'
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'loss': last_loss, # Or the average loss for the epoch
        }, checkpoint_path)
    except Exception as e:
        print("Could not save checkpoint -> error: ", e)
    check_val_perf(net, test_loader, wall_criterion, room_criterion, device, writer, global_step)
    
    writer.close()
    #Saving as plain parameter checkpoint
    torch.save(net.state_dict(), '../models/test_model.pth')
    print("Complete - saved trained model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a floorplan segmentation model.')
    parser.add_argument('--resume-epoch', type=int, default=None,
                        help='Epoch number to resume training from.')
    
    args = parser.parse_args()
    torch.cuda.empty_cache()
    train(resume_from_epoch=args.resume_epoch)