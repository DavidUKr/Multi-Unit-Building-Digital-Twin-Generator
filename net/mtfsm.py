import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# from net.scg_gcn import SCG_block, GCN_Layer
from scg_gcn import SCG_block, GCN_Layer

#Multi-Taks Floorplan Segmentation Model

class MTFSM(nn.Module):

    def __init__(self, wall_num_classes, room_num_classes, pretrained_encoder=False, scg_node_size=(32,32),
                 dropout=0, enhance_diag=True, aux_pred=True):

        super(MTFSM, self).__init__()

        self.wall_num_classes=wall_num_classes
        self.room_num_classes=room_num_classes
        self.aux_pred=aux_pred
        self.node_size=scg_node_size

        resnet101=resnet101 = models.resnet101(weights='DEFAULT' if pretrained_encoder else None)
        #input layer
        self.conv1=resnet101.conv1 #7x7, stride 2
        self.bn1=resnet101.bn1
        self.relu=resnet101.relu
        self.maxpool=resnet101.maxpool #3x3, stride 2
        #conv2->conv5
        self.conv2=resnet101.layer1 #256 channels
        self.conv3=resnet101.layer2 #512 channels
        self.conv4=resnet101.layer3 #1024 channels
        self.conv5=resnet101.layer4 #2048 channels

        #Decoder
        #WallNet
        self.upsample1=nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec1=nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.upsample2=nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec2=nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.upsample3=nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3=nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upsample4=nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec4=nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upsample5=nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv=nn.Conv2d(64, self.wall_num_classes, kernel_size=1)

        #Todo: fix modele inst parameters

        #RoomNet
        self.graph_layers1 = GCN_Layer(2048, 256, bnorm=True, activation=nn.ReLU(True), dropout=dropout)

        self.graph_layers2 = GCN_Layer(256, self.room_num_classes, bnorm=False, activation=None)

        self.scg = SCG_block(in_ch=2048,
                             hidden_ch=self.room_num_classes,
                             node_size=scg_node_size,
                             add_diag=enhance_diag,
                             dropout=dropout)

    def forward(self, x):
        x_size=x.size()
        #Encoder
        x1=self.conv1(x)
        x1=self.bn1(x1)
        x1=self.relu(x1)
        x1_pool=self.maxpool(x1)

        x2=self.conv2(x1_pool)
        x3=self.conv3(x2)
        x4=self.conv4(x3)
        x5=self.conv5(x4)

        #WallNet
        d1=self.upsample1(x5)
        d1=torch.concat([d1,x4], dim=1)
        d1=self.dec1(d1)

        d2=self.upsample2(d1)
        d2=torch.concat([d2,x3], dim=1)
        d2=self.dec2(d2)

        d3=self.upsample3(d2)
        d3=torch.concat([d3,x2], dim=1)
        d3=self.dec3(d3)

        d4=self.upsample4(d3)
        d4=torch.concat([d4,x1], dim=1)
        d4=self.dec4(d4)

        d5=self.upsample5(d4)
        wall_out=self.final_conv(d5)

        #RoomNet
        B, C, H, W = x5.size()

        A, gx, room_loss, z_hat = self.scg(x5)
        gx, _ = self.graph_layers2(self.graph_layers1((gx.view(B, -1, C), A)))
        if self.aux_pred:
            gx += z_hat
        gx = gx.view(B, self.room_num_classes, self.node_size[0], self.node_size[1])
        gx = F.interpolate(gx, (H, W), mode='bilinear', align_corners=False)

        room_out=F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False)

        return wall_out, room_out, A, room_loss
    
def get_model(wall_num_classes=4, room_num_classes=7, pretrained_encoder=False, dropout=0):
    return MTFSM(wall_num_classes=wall_num_classes, room_num_classes=room_num_classes, pretrained_encoder=pretrained_encoder, dropout=dropout)