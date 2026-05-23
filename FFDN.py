import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 3, 1)
        self.cv2 = Conv(hidden, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.blocks = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(block(y[-1]) for block in self.blocks)
        return self.cv2(torch.cat(y, dim=1))


class Adown(nn.Module):
    # Adaptive down-sampling used inside the Focus unit

    def __init__(self, channels):
        super().__init__()
        self.alpha_logits = nn.Parameter(torch.zeros(4, channels))
        self.beta = nn.Parameter(torch.zeros(1))
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, but got shape {tuple(x.shape)}")

        b, c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2))
            b, c, h, w = x.shape

        patches = torch.stack(
            (
                x[:, :, 0::2, 0::2],
                x[:, :, 1::2, 0::2],
                x[:, :, 0::2, 1::2],
                x[:, :, 1::2, 1::2],
            ),
            dim=2,
        )

        alpha = self.alpha_logits.softmax(dim=0).t().view(1, c, 4, 1, 1)
        return (patches * alpha).sum(dim=2) + self.beta * self.avg_pool(x)


class DSConv(nn.Module):
    # Depthwise separable convolution

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.dw = Conv(channels, channels, kernel_size, 1, g=channels)
        self.pw = Conv(channels, channels, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class FocusUnit(nn.Module):
    def __init__(self, c_large, c_mid, c_small, out_channels, kernels=(5, 7, 9, 11)):
        super().__init__()

        self.large_proj = Conv(c_large, out_channels, 1, 1)
        self.mid_proj = Conv(c_mid, out_channels, 1, 1)
        self.small_proj = Conv(c_small, out_channels, 1, 1)

        self.adown = Adown(out_channels)

        fusion_channels = out_channels * 3
        self.branches = nn.ModuleList(
            DSConv(fusion_channels, k) for k in kernels
        )
        self.compress = Conv(fusion_channels, out_channels, 1, 1)

    def _align_large_to_mid(self, x_large, target_size):
        if x_large.shape[-2:] == target_size:
            return x_large

        h, w = x_large.shape[-2:]
        th, tw = target_size

        if h == th * 2 and w == tw * 2:
            return self.adown(x_large)

        raise ValueError(
            f"x_large should have the same size as x_mid or be 2x larger, "
            f"but got {x_large.shape[-2:]} and {target_size}."
        )

    def forward(self, x_large, x_mid, x_small):
        large = self.large_proj(x_large)
        mid = self.mid_proj(x_mid)
        small = self.small_proj(x_small)

        target_size = mid.shape[-2:]
        large = self._align_large_to_mid(large, target_size)

        if small.shape[-2:] != target_size:
            small = F.interpolate(small, size=target_size, mode="nearest")

        x = torch.cat((large, mid, small), dim=1)

        y = sum(branch(x) for branch in self.branches)
        y = x + y

        return self.compress(y)


class FFDN(nn.Module):
    # Feature-Focused Diffusion Network

    def __init__(self, channels, c2f_repeats=1):
        super().__init__()

        if len(channels) != 3:
            raise ValueError("channels should be a tuple/list like (c1, c2, c3).")

        c1, c2, c3 = channels

        self.focus1 = FocusUnit(c1, c2, c3, c2)
        self.focus2 = FocusUnit(c1, c2, c3, c2)

        self.up1_c2f = C2f(c1 + c2, c1, n=c2f_repeats)
        self.down1 = Conv(c2, c3, 3, 2)
        self.down1_c2f = C2f(c3 + c3, c3, n=c2f_repeats)

        self.up2_c2f = C2f(c1 + c2, c1, n=c2f_repeats)
        self.down2 = Conv(c2, c3, 3, 2)
        self.down2_c2f = C2f(c3 + c3, c3, n=c2f_repeats)

    def forward(self, features):
        if len(features) != 3:
            raise ValueError(f"Expected three feature maps, but got {len(features)}.")

        p1, p2, p3 = features

        f1 = self.focus1(p1, p2, p3)

        up1 = F.interpolate(f1, size=p1.shape[-2:], mode="nearest")
        up1 = self.up1_c2f(torch.cat((up1, p1), dim=1))

        down1 = self.down1(f1)
        down1 = self.down1_c2f(torch.cat((down1, p3), dim=1))

        f2 = self.focus2(up1, f1, down1)

        p4 = F.interpolate(f2, size=up1.shape[-2:], mode="nearest")
        p4 = self.up2_c2f(torch.cat((p4, up1), dim=1))

        p5 = f2

        p6 = self.down2(f2)
        p6 = self.down2_c2f(torch.cat((p6, down1), dim=1))

        return p4, p5, p6