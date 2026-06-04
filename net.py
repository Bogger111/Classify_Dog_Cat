import torch
import torch.nn as nn

class MyResNetBlock(nn.Module):
    def __init__(self,input_channel,output_channel,stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=input_channel,out_channels=output_channel,
            kernel_size=3,padding=1,stride=stride
            )
        self.bn1 = nn.BatchNorm2d(output_channel)

        self.conv2 = nn.Conv2d(
            in_channels=output_channel,out_channels=output_channel,
            kernel_size=3,padding=1,stride=1
        )
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.relu = nn.ReLU()

        if stride != 1 or input_channel != output_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channel,out_channels=output_channel,
                    kernel_size=1,stride=stride
                ),
                nn.BatchNorm2d(output_channel)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self,x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = out + identity
        out = self.relu(out)

        return out

class MyResNet18(nn.Module):
    def __init__(self,num_classes,input_channel=3,output_channel=64,dropout=0):
        super().__init__()

        self.channel = output_channel
        self.reconv = nn.Sequential(
            nn.Conv2d(
            in_channels=input_channel,out_channels=output_channel,
            kernel_size=3,padding=1,stride=1
            ),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(64,2,stride=1)
        self.layer2 = self._make_layer(128,2,stride=2)
        self.layer3 = self._make_layer(256,2,stride=2)
        self.layer4 = self._make_layer(512,2,stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        
        self.fc = nn.Linear(self.channel,num_classes)

    def _make_layer(self,output_channel,depth,stride):
        layers = []

        layers.append(MyResNetBlock(self.channel,output_channel,stride))
        self.channel = output_channel

        for _ in range(1,depth):
            layers.append(MyResNetBlock(self.channel,output_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.reconv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc(x)

        return x
    
