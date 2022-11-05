from torch import nn
from torch import cat

class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = out_channels if in_channels > out_channels else out_channels//2

        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm3d(inter_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)


class unet3dEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dEncoder, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pub(x)
        return x,self.pool(x)


class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub = pub(in_channels//2+in_channels, out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)
        #c1 = (x1.size(2) - x.size(2)) // 2
        #c2 = (x1.size(3) - x.size(3)) // 2
        #x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        x = cat((x, x1), dim=1)
        x = self.pub(x)
        return x


class unet3d(nn.Module):
    def __init__(self, args):
        super(unet3d, self).__init__()
        init_channels = 4
        class_nums = 3
        batch_norm = True
        sample = True

        self.en1 = unet3dEncoder(init_channels, 64, batch_norm)
        self.en2 = unet3dEncoder(64, 128, batch_norm)
        self.en3 = unet3dEncoder(128, 256, batch_norm)
        self.en4 = unet3dEncoder(256, 512, batch_norm)

        self.up3 = unet3dUp(512, 256, batch_norm, sample)
        self.up2 = unet3dUp(256, 128, batch_norm, sample)
        self.up1 = unet3dUp(128, 64, batch_norm, sample)
        self.con_last = nn.Conv3d(64, class_nums, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1,x = self.en1(x)
        x2,x= self.en2(x)
        x3,x= self.en3(x)
        x4,_ = self.en4(x)

        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.con_last(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()