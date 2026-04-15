import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale
# Double Convolutional layer at each encoder layer
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x 
# Attention Gate Structure
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        # g: gating signal (from decoder), x: skip connection
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi
    
class Att_H_UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature=[32,64,128,256]):
        super().__init__()
        self.downs = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(feature[0], out_channels, kernel_size=1)

        self.bottleneck = nn.Sequential(
            DoubleConv(feature[3], feature[3] * 2),
            nn.Conv2d(feature[3]*2, feature[3], 3, 1, 1, bias=False)
        ) 
        
        self.se = nn.ModuleList([
            SEBlock(feature[i]) for i in range(4)
        ])
        # Encoder loop
        for i in range(4):
            self.downs.append(DoubleConv(in_channels,feature[i]))
            in_channels = feature[i]
        # For Upsampling
        self.trans = nn.ModuleList([
            nn.ConvTranspose2d(feature[i] , feature[i-1], 2, 2) for i in range(3, 0, -1)
        ])
        self.tran = nn.ConvTranspose2d(feature[3] , feature[3], 2, 2)

        self.att = nn.ModuleList([
            AttentionGate(F_g=feature[i], F_l=feature[i], F_int=feature[i]//2) for i in range(3, -1, -1)
        ])


    def forward(self, x):
        skip_connections = []

        for num, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(self.se[num](x))
            x = self.pool(x)
        x = self.bottleneck(x)
        x = self.tran(x)
        for ind, skip in enumerate(reversed(skip_connections)):
            

            if x.shape[2:] != skip.shape[2:]:
                x = TF.resize(x, size=skip.shape[2:])
            att_skip = self.att[ind](g=x, x=skip)
            x = x + att_skip

            if not ind == 3:
                x = self.trans[ind](x)

        return self.final_conv(x)
    
def test():
    x = torch.randn((3, 3, 161, 161))
    model = Att_H_UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total number of parameters: {total_params}")

if __name__ == "__main__":
    test()
    