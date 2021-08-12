import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ResidualModule(nn.Layer):
    def __init__(self, in_channels=256):
        super().__init__()
        self.bn = nn.BatchNorm2D(num_features=in_channels)
        self.conv1 = nn.Conv2D(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.bn(x))
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = self.conv3(y)
        return x + y


class HourglassModule(nn.Layer):
    def __init__(self, heatmap_size=64, in_channels=256):
        # hourglass模块的shape保证为(N, 64, 64, 256)
        # TODO: 这个操作挺奇葩
        super().__init__()
        self.downsamples = nn.Sequential(
            *[(f'down_{i}', ResidualModule()) for i in range(4)]
        )

        self.keeps = nn.Sequential(
            *[(f'keep_{i}', ResidualModule()) for i in range(4)]
        )

        self.low = nn.Sequential(
            *[(f'low_{i}', ResidualModule()) for i in range(3)]
        )

    def forward(self, x):
        temp = []

        # 64 -> 32 -> 16 -> 8 -> 4
        for down, keep in zip(self.downsamples, self.keeps):
            x = down(x)
            temp.append(keep(x))
            x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # 4x4
        x = self.low(x)

        # 4 -> 8 -> 16 -> 32 -> 64
        for feat in temp[::-1]:
            x = F.upsample(x, size=(2, 2), mode='nearest')
            x = x + feat

        return x


if __name__ == '__main__':
    hg = HourglassModule()

    x = paddle.randn((4, 3, 256, 256))
    y = hg(x)
    print(y.shape)
