import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format
import time


class SECA(nn.Module):  # Spatial-Efficient Channel Attention (SECA)
    def __init__(self, channel, reduction=8):  ###16 or 8(mini)
        super(SECA, self).__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # spacial attn:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_attention(spatial_weight)

        # channel attn (2 MLP)
        channel_weight = self.channel_attention(x)

        out = x * spatial_weight * channel_weight
        return out


class Conv(nn.Module):
    def __init__(self, N):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N, N * 2, 1),
            nn.BatchNorm2d(N * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N * 2, N, 3, padding=1),
            nn.BatchNorm2d(N),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class FFN(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(N, N * 2, 1),
            nn.BatchNorm2d(N * 2),
            nn.GELU(),
            nn.Conv2d(N * 2, N, 1),
            nn.BatchNorm2d(N)
        )

    def forward(self, x):
        return self.ffn(x) + x


class Attn(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.token_mixer = Conv(N)
        self.channel_mixer = FFN(N)
        self.attn = SECA(N, reduction=8)
        self.norm1 = nn.BatchNorm2d(N)
        self.norm2 = nn.BatchNorm2d(N)

    def forward(self, x):
        out = self.token_mixer(x)
        out = self.norm1(out)
        out = self.attn(out)
        out = self.channel_mixer(out)
        out = self.norm2(out)
        out += x  # 添加残差连接
        return out


class RTSR(nn.Module):
    def __init__(self, sr_rate=4, N=8):  ### 16 or 8: mini
        super(RTSR, self).__init__()
        self.scale = sr_rate

        self.head = nn.Sequential(
            nn.Conv2d(3, N, 3, padding=1),
            nn.BatchNorm2d(N),
            nn.ReLU(inplace=True)
        )

        self.body = nn.Sequential(
            *[Attn(N) for _ in range(4)]
        )

        self.tail = nn.Sequential(
            nn.Conv2d(N, 3 * sr_rate * sr_rate, 1),
            nn.PixelShuffle(sr_rate)
        )

    def forward(self, x):
        head = self.head(x)

        body_out = head
        for attn_layer in self.body:
            body_out = attn_layer(body_out)

        h = self.tail(body_out)

        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        out = h + base
        # out = base

        return out


if __name__ == '__main__':
    model = RTSR(sr_rate=4, N=8)  ### 16, 8

    # Params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {params}')

    # GFLOPs
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f'GFLOPs: {macs}')

    # runtime
    input_tensor = torch.randn(1, 3, 224, 224).cuda()
    model = model.cuda()
    start_time = time.time()
    output = model(input_tensor)
    end_time = time.time()
    runtime_ms = (end_time - start_time) * 1000
    print(f'Runtime (ms): {runtime_ms}')