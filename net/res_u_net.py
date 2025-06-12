import torch
import torch.nn as nn
import torchvision.models as models

class ResNet101UNet(nn.Module):

    def __init__(self, num_classes, pretrained=False):

        super(ResNet101UNet, self).__init__()

        resnet101=models.resnet101(pretrained=pretrained)
        #Encoder
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
        self.final_conv=nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        #Encoder
        x1=self.conv1(x)
        x1=self.bn1(x1)
        x1=self.relu(x1)
        x1_pool=self.maxpool(x1)

        x2=self.conv2(x1_pool)
        x3=self.conv3(x2)
        x4=self.conv4(x3)
        x5=self.conv5(x4)

        #Decoder
        d1=self.upsample1(x5)
        d1=torch.concat([d1,x4], dim=1)
        d1=self.dec1(d1)

        d2=self.upsample1(d1)
        d2=torch.concat([d2,x3], dim=1)
        d2=self.dec1(d2)

        d3=self.upsample1(d2)
        d3=torch.concat([d3,x2], dim=1)
        d3=self.dec1(d3)

        d4=self.upsample1(d3)
        d4=torch.concat([d4,x1], dim=1)
        d4=self.dec1(d4)

        d5=self.upsample5(d4)
        out=self.final_conv(d5)

        return out
    
def get_model(num_classes=3, pretrained=False):
    return ResNet101UNet(num_classes=num_classes, pretrained=pretrained)