import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, groups=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AWD(nn.Module):
    # Adaptive-Weighted Down-sampling

    def __init__(self, channels, kernel_size=3, groups=4, attn_kernel=3):
        super().__init__()

        if channels % groups != 0:
            raise ValueError(f"channels={channels} must be divisible by groups={groups}")
        if kernel_size % 2 == 0 or attn_kernel % 2 == 0:
            raise ValueError("kernel_size and attn_kernel should be odd")

        self.channels = channels
        self.num_candidates = 4

        self.attn_pool = nn.AvgPool2d(
            kernel_size=attn_kernel,
            stride=1,
            padding=attn_kernel // 2,
            count_include_pad=False,
        )
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.down = nn.Conv2d(
            channels,
            channels * self.num_candidates,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )

    @staticmethod
    def _pad_even(x):
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        return x

    @staticmethod
    def _split_2x2(x):
        b, c, h, w = x.shape
        h2, w2 = h // 2, w // 2

        x = x.reshape(b, c, h2, 2, w2, 2)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        return x.reshape(b, c, h2, w2, 4)

    def _split_candidates(self, x, h2, w2):
        b = x.shape[0]

        x = x.reshape(b, self.num_candidates, self.channels, h2, w2)
        return x.permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, x):
        x = self._pad_even(x)

        h2 = x.shape[-2] // 2
        w2 = x.shape[-1] // 2

        weight = self.attn(self.attn_pool(x))
        weight = self._split_2x2(weight).softmax(dim=-1)

        feat = self.down(x)
        feat = self._split_candidates(feat, h2, w2)

        return (feat * weight).sum(dim=-1)


class AWDConv(nn.Module):

    def __init__(
        self,
        c1,
        c2,
        kernel_size=3,
        groups=4,
        attn_kernel=3,
        act=True,
        use_projection=True,
    ):
        super().__init__()

        self.awd = AWD(
            channels=c1,
            kernel_size=kernel_size,
            groups=groups,
            attn_kernel=attn_kernel,
        )

        if use_projection or c1 != c2:
            self.proj = ConvBNAct(c1, c2, k=1, s=1, act=act)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(self.awd(x))


if __name__ == "__main__":
    x = torch.randn(1, 64, 640, 640)

    awd = AWD(64)
    y = awd(x)
    print("AWD:", y.shape)
    
    awd_conv = AWDConv(64, 128)
    y = awd_conv(x)
    print("AWDConv:", y.shape)