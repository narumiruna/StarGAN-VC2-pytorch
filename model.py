from torch import nn
import torch


class ConvINGLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(ConvINGLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride=stride,
                      padding=padding),
            nn.InstanceNorm2d(out_planes),
            nn.GLU(dim=1),
        )


class ConditionalInstanceNorm1d(nn.Module):
    def __init__(self, in_dim, out_dim, eps=1e-5):
        super(ConditionalInstanceNorm1d, self).__init__()
        self.eps = eps
        self.gamma = nn.Embedding(in_dim, out_dim)
        self.beta = nn.Embedding(in_dim, out_dim)

    def forward(self, x, c):
        feat_mean = x.mean(dim=2, keepdim=True)
        feat_std = x.var(dim=2, keepdim=True).add(self.eps).sqrt()

        x = x.sub(feat_mean).div(feat_std)

        gamma = self.gamma(c).view(x.size(0), -1, 1)
        beta = self.beta(c).view(x.size(0), -1, 1)

        return gamma * x + beta


class Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 code_dim=8):
        super(Block, self).__init__()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding)
        self.norm = ConditionalInstanceNorm1d(code_dim, out_channels)
        self.glu = nn.GLU(dim=1)

    def forward(self, x, c):
        x = self.conv(x)
        x = self.norm(x, c)
        x = self.glu(x)
        return x


class ConvPSGLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(ConvPSGLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride=stride,
                      padding=padding),
            nn.PixelShuffle(2),
            nn.GLU(dim=1),
        )


class Generator(nn.Module):
    def __init__(self, code_dim=8):
        super(Generator, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 128, (5, 15), stride=1, padding=(2, 7)),
            nn.GLU(dim=1),
        )
        self.conv_2 = ConvINGLU(64, 256, 5, stride=2, padding=2)
        self.conv_3 = ConvINGLU(128, 512, 5, stride=2, padding=2)
        self.conv_4 = nn.Sequential(
            nn.Conv1d(2304, 256, 1),
            nn.InstanceNorm1d(256),
        )

        self.blocks = nn.ModuleList([
            Block(256, 512, 5, padding=2, code_dim=code_dim) for _ in range(9)
        ])

        self.conv_5 = nn.Conv1d(256, 2304, 1, stride=1)
        self.conv_6 = ConvPSGLU(256, 1024, 5, padding=2)
        self.conv_7 = ConvPSGLU(128, 512, 5, padding=2)
        self.conv_8 = nn.Conv2d(64, 1, 1)

    def forward(self, x, c):
        batch_size, height, width = x.size()
        x = x.unsqueeze(1)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(batch_size, -1, width // 4)
        x = self.conv_4(x)

        for block in self.blocks:
            x = block(x, c)

        x = self.conv_5(x)
        x = x.view(batch_size, 256, height // 4, width // 4)

        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.conv_8(x)

        x = x.squeeze(1)
        return x


def test_generator():
    batch_size = 32
    h = 36
    w = 128
    code_dim = 8
    x = torch.randn(batch_size, h, w)
    c = torch.randint(low=0, high=code_dim - 1, size=(batch_size, ))

    model = Generator(code_dim)
    y = model(x, c)
    assert len(y.size()) == [batch_size, h, w]
