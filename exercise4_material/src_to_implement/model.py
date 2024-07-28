import torch.nn as nn
from torch.nn.modules.flatten import Flatten



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

        self.use_conv1x1 = in_channels != out_channels or stride != 1
        if self.use_conv1x1:
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
            self.batch_norm3 = nn.BatchNorm2d(num_features=out_channels)
        
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        main_output = self.main_path(x)
        if self.use_conv1x1:
            shortcut_output = self.conv1x1(x)
            shortcut_output = self.batch_norm3(shortcut_output)
        else:
            shortcut_output = x
            
        final_output = self.act(main_output + shortcut_output)
        return final_output


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet(x)
