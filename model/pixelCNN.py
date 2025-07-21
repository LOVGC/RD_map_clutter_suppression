class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer("mask", torch.ones_like(self.weight))
        
        _, _, kH, kW = self.weight.size()
        yc, xc = kH // 2, kW // 2

        # 创建 mask
        self.mask[:, :, yc+1:, :] = 0
        self.mask[:, :, yc, xc+1:] = 0
        if mask_type == 'A':
            self.mask[:, :, yc, xc] = 0  # 不看当前像素

    def forward(self, x):
        self.weight.data *= self.mask  # apply mask
        return super().forward(x)



class PixelCNN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=64, kernel_size=7, num_layers=7):
        super().__init__()
        layers = []

        # 第一层用 mask A（不看当前像素）
        layers.append(MaskedConv2d('A', input_channels, hidden_channels, kernel_size, padding=kernel_size//2))
        layers.append(nn.ReLU())

        # 后面几层用 mask B（允许当前像素，但不看未来）
        for _ in range(num_layers - 2):
            layers.append(MaskedConv2d('B', hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())

        # 输出层：预测每个像素 256 个可能的值
        layers.append(MaskedConv2d('B', hidden_channels, 256, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 输入为 (B, 1, H, W)，值范围 0~255
        return self.net(x)


