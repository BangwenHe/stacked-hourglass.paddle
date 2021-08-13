import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ResidualModule(nn.Layer):
    def __init__(self, in_channels=256):
        super().__init__()
        self.bn = nn.BatchNorm(num_channels=in_channels)
        self.conv1 = nn.Conv2D(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.bn(x))
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = self.conv3(y)
        return x + y


class HourglassModule(nn.Layer):
    def __init__(self, heatmap_size=64, in_channels=256):
        # hourglass模块的shape保证为(N, 256, 64, 64)
        super().__init__()
        self.downsamples = nn.LayerList(
            [ResidualModule() for _ in range(4)]
        )

        self.keeps = nn.LayerList(
            [ResidualModule() for _ in range(4)]
        )

        self.low = nn.Sequential(
            *[ResidualModule() for _ in range(3)]
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
            x = F.upsample(x, size=feat.shape[-2:], mode='nearest')
            x = x + feat

        return x


class Hourglass(nn.Layer):
    def __init__(self, num_modules=4, img_size=256, feat_size=64, num_channels=256, num_joints=16):
        super().__init__()

        # 堆叠的hourglass模块的数量
        self.num_modules = num_modules
        # 输入网络的图像尺寸
        self.img_size = img_size
        # 输入hourglass模块的图像尺寸
        self.feat_size = feat_size
        # 输入hourglass模块的通道数
        self.num_channels = num_channels
        # 图片的通道数
        self.num_joints = num_joints

        # 对256的图像进行降采样
        self.down_sample = nn.Sequential(
            nn.Conv2D(kernel_size=7, stride=2, padding=3, in_channels=3, out_channels=num_channels),
            ResidualModule(in_channels=num_channels),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

        self.hourglass_modules = nn.LayerList()
        for i in range(self.num_modules):
            module = nn.Sequential(
                ('hourglass_module', HourglassModule()),
                ('remap_joints', nn.Conv2D(kernel_size=1, in_channels=self.num_channels, out_channels=self.num_joints)),
                ('remap_channels_1', nn.Conv2D(kernel_size=1, in_channels=self.num_joints, out_channels=self.num_channels)),
                ('remap_channels_2', nn.Conv2D(kernel_size=1, in_channels=self.num_joints, out_channels=self.num_channels))
            )

            self.hourglass_modules.append(module)

    def forward(self, x):
        x = self.down_sample(x)
        output = []

        for idx, m in enumerate(self.hourglass_modules):
            _hg = m['hourglass_module']
            rj = m['remap_joints']
            r1 = m['remap_channels_1']
            r2 = m['remap_channels_2']
            y = _hg(x)
            y = rj(y)
            output.append(y)

            y1 = r1(y)
            y2 = r2(y)

            x = x + y1 + y2

        return output

    def init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0]*m.weight.shape[1]*m.weight.shape[2]
                v = np.random.normal(loc=0.,scale=np.sqrt(2./n),size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))


if __name__ == '__main__':
    hg = Hourglass()
    hg.init_weight()
    print(hg)

    x = paddle.randn((4, 3, 256, 256))
    y = hg(x)
    print(y[0].shape, y[1].shape)

    paddle.save(hg.state_dict(), 'hg.pdparams')
