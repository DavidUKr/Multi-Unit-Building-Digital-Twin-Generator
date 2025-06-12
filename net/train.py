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


#Hyperparameters
alpha=0.5
lr=5e-4
weight_decay=2e-5
batch_size =10
num_epochs =2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading model")
net = model.get_model(pretrained_encoder=True, dropout=0.2)
net= net.to(device)
print('Loading processes to device')
optimizer=optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
wall_criterion=ACW_loss().to(device)
room_criterion=ACW_loss().to(device)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #for resnet pretrained
])
print("Loading dataset")
train_dataset= FloorplanDataset(dataset_dir="../train_data/mufp_10", split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_dataset= FloorplanDataset(dataset_dir="../train_data/mufp_10", split='test', transform=transform)
test_loader= DataLoader(test_dataset, batch_size=1, shuffle=False)

def validate_model(net, val_loader, wall_criterion, room_criterion, device, D=0.5):
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
            wall_correct += (wall_preds == wall_masks).sum().item()
            wall_total += wall_masks.numel()
            room_correct += (room_preds == room_masks).sum().item()
            room_total += room_masks.numel()

    avg_val_loss = val_loss / len(val_loader)
    wall_iou = wall_iou_metric.compute().item()
    room_iou = room_iou_metric.compute().item()
    wall_accuracy = wall_correct / wall_total
    room_accuracy = room_correct / room_total

    net.train()  # Restore training mode
    return avg_val_loss, wall_iou, room_iou, wall_accuracy, room_accuracy

def check_val_perf(test_net, summary_writer=None, global_step=None, verbose=True):
    print("Validation performance")
    avg_val_loss, wall_iou, room_iou, wall_accuracy, room_accuracy=validate_model(test_net, test_loader, wall_criterion, room_criterion, device, D=0.5)

    if summary_writer and global_step:
        summary_writer.add_scalar('avg_val_loss', avg_val_loss, global_step)
        summary_writer.add_scalar("wall_iou:", wall_iou, global_step)
        summary_writer.add_scalar("room_iou:", room_iou, global_step)
        summary_writer.add_scalar("wall_accuracy:", wall_accuracy, global_step)
        summary_writer.add_scalar("room_accuracy:", room_accuracy, global_step)

    if verbose:
        print("avg_val_loss:", avg_val_loss)
        print("wall_iou:", wall_iou)
        print("room_iou:", room_iou)
        # print("wall_accuracy:", wall_accuracy)
        # print("room_accuracy:", room_accuracy)

#Train loop
def train():
    print("Start training")
    writer = SummaryWriter("runs/mtfsm")
    # writer.add_graph(net, train_dataset.__getitem__(0)[0].unsqueeze(0))
    global_step = 0
    losses=[]
    steps=[]

    net.train()

    for epoch in range(num_epochs):
        running_loss=0.0
        last_loss=0.0
        epoch_step=0

        # for i, (images, wall_masks, room_masks) in enumerate(train_loader):
        for i, (images, wall_masks, room_masks) in enumerate(train_loader):
            images=images.to(device)
            wall_masks=wall_masks.to(device)
            room_masks=room_masks.to(device)
            #fw pass
            wall_out, room_out, graph_out, room_scg_loss=net(images)
            #loss
            wall_loss=wall_criterion(wall_out, wall_masks)
            room_loss=room_scg_loss+room_criterion(room_out, room_masks)
            loss=alpha*room_loss + (1-alpha)*wall_loss
            print(f'Step {i} losses:', 'wall:', wall_loss.item(),'room:', room_loss.item(),'total:', loss.item())

            #Visualization
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            losses.append(loss.item())
            steps.append(global_step)
            global_step += 1
            epoch_step += 1
            running_loss+= loss.item()
            last_loss= loss.item()
            
            #backward
            loss.backward()
            optimizer.step()

        
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/epoch_step:.4f} Last_Loss: {last_loss}')
        
        torch.save(net.state_dict(), f'checkpoints/trained_model_ep{epoch}.pth')
        check_val_perf(net, summary_writer=writer, global_step=global_step)

    writer.close()
    
    torch.save(net.state_dict(), '../models/test_model.pth')
    print("Complete - saved trained model")

if __name__ == "__main__":
    train()