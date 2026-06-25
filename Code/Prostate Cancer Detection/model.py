import torch
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )

class Encoder3D(nn.Module):
    """
    U-Net encoder trunk -> output feature map.
    """
    def __init__(self, in_channels=1, base=32):
        super().__init__()
        self.e1 = conv_block(in_channels, base)
        self.p1 = nn.MaxPool3d(2)
        self.e2 = conv_block(base, base*2)
        self.p2 = nn.MaxPool3d(2)
        self.e3 = conv_block(base*2, base*4)
        self.p3 = nn.MaxPool3d(2)
        self.b  = conv_block(base*4, base*8)
        self.out_channels = base * 8

    def forward(self, x):
        x = self.e1(x)
        x = self.e2(self.p1(x))
        x = self.e3(self.p2(x))
        x = self.b(self.p3(x))
        return x

class ModalityNet(nn.Module):
    def __init__(self, in_channels=1, base=32, feat_dim=256, dropout=0.1):
        super().__init__()
        self.enc = Encoder3D(in_channels=in_channels, base=base)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.enc.out_channels, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        fm = self.enc(x)
        z = self.head(self.pool(fm))
        return z  # (B,feat_dim)

class ThreeModalityClassifier(nn.Module):
    def __init__(self, base=32, feat_dim=256, dropout=0.1, fusion_hidden=256):
        super().__init__()
        self.net0 = ModalityNet(in_channels=1, base=base, feat_dim=feat_dim, dropout=dropout)
        self.net1 = ModalityNet(in_channels=1, base=base, feat_dim=feat_dim, dropout=dropout)
        self.net2 = ModalityNet(in_channels=1, base=base, feat_dim=feat_dim, dropout=dropout)

        self.fusion = nn.Sequential(
            nn.Linear(3 * feat_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_tuple):
        x0, x1, x2 = x_tuple
        z0 = self.net0(x0)
        z1 = self.net1(x1)
        z2 = self.net2(x2)
        z = torch.cat([z0, z1, z2], dim=1)
        logit = self.fusion(z)  # (B,1)
        return logit
