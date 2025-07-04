
import torch.nn as nn

class LKDA(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 输入 RGB 图像
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)   # 恢复到原始通道数
        )

    def forward(self, x):
        return self.features(x)
