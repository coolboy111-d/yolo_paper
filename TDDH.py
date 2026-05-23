import torch
import torch.nn as nn

try:
    from torchvision.ops import DeformConv2d
except Exception:
    DeformConv2d = None


def make_gn_groups(channels, groups=32):
    groups = min(groups, channels)
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return groups


class ConvGN(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, groups=32, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.gn = nn.GroupNorm(make_gn_groups(c2, groups), c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class Scale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.weight + self.bias


class TaskDecomposition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvGN(in_channels, out_channels, k=1),
            ConvGN(out_channels, out_channels, k=3),
        )

    def forward(self, x):
        return self.block(x)


class DynamicFilter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.filter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.filter(x)


class DCNv2Block(nn.Module):
    def __init__(self, channels, guide_channels, kernel_size=3):
        super().__init__()

        if DeformConv2d is None:
            raise ImportError("DCNv2Block requires torchvision.ops.DeformConv2d.")

        padding = kernel_size // 2
        offset_channels = 2 * kernel_size * kernel_size
        mask_channels = kernel_size * kernel_size

        self.generator = nn.Conv2d(
            guide_channels,
            offset_channels + mask_channels,
            kernel_size=3,
            padding=1,
        )
        self.dcn = DeformConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(make_gn_groups(channels), channels)
        self.act = nn.SiLU(inplace=True)
        self.offset_channels = offset_channels

    def forward(self, x, guide):
        offset_mask = self.generator(guide)
        offset = offset_mask[:, :self.offset_channels]
        mask = offset_mask[:, self.offset_channels:].sigmoid()

        x = self.dcn(x, offset, mask)
        return self.act(self.norm(x))


class TDDH(nn.Module):
    # Task-Aligned Dynamic Detection Head

    def __init__(self, in_channels, hidden_channels, num_classes, reg_max=16):
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError("in_channels should contain three levels, e.g. (c4, c5, c6).")

        self.in_channels = tuple(in_channels)
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.reg_channels = 4 * reg_max
        self.num_levels = len(in_channels)

        self.input_proj = nn.ModuleList(
            nn.Identity() if c == hidden_channels else ConvGN(c, hidden_channels, k=1)
            for c in in_channels
        )

        self.shared_conv = nn.Sequential(
            ConvGN(hidden_channels, hidden_channels, k=3),
            ConvGN(hidden_channels, hidden_channels, k=3),
        )

        inter_channels = hidden_channels * 2

        self.loc_decomp = TaskDecomposition(inter_channels, hidden_channels)
        self.cls_decomp = TaskDecomposition(inter_channels, hidden_channels)

        self.dcn = DCNv2Block(hidden_channels, inter_channels, kernel_size=3)
        self.dynamic_filter = DynamicFilter(inter_channels, hidden_channels)

        self.bbox_preds = nn.ModuleList(
            nn.Conv2d(hidden_channels, self.reg_channels, kernel_size=1)
            for _ in range(self.num_levels)
        )
        self.cls_preds = nn.ModuleList(
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
            for _ in range(self.num_levels)
        )
        self.scales = nn.ModuleList(
            Scale(self.reg_channels) for _ in range(self.num_levels)
        )

    def forward(self, features):
        if len(features) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature maps, but got {len(features)}.")

        outputs = []

        for i, x in enumerate(features):
            if x.shape[1] != self.in_channels[i]:
                raise ValueError(
                    f"Level {i}: expected {self.in_channels[i]} channels, got {x.shape[1]}."
                )

            x = self.input_proj[i](x)

            shared = self.shared_conv(x)
            inter = torch.cat((shared, x), dim=1)

            loc_feat = self.loc_decomp(inter)
            cls_feat = self.cls_decomp(inter)

            loc_feat = self.dcn(loc_feat, inter)
            cls_feat = cls_feat * self.dynamic_filter(inter)

            bbox = self.scales[i](self.bbox_preds[i](loc_feat))
            cls = self.cls_preds[i](cls_feat)

            outputs.append(torch.cat((bbox, cls), dim=1))

        return outputs