import paddle
import paddle.nn as nn


class JointsMSELoss(nn.Layer):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(num_or_sections=num_joints, axis=1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(num_or_sections=num_joints, axis=1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


if __name__ == '__main__':
    from lib.models.hourglass import Hourglass

    x = paddle.randn((4, 3, 256, 256))
    label = paddle.randn((4, 17, 64, 64))

    hg = Hourglass()
    criterion = JointsMSELoss(False)
    optim = paddle.optimizer.Adam(learning_rate=1e-3, parameters=hg.parameters())
    print(hg.down_sample[0].weight[0, 0])

    y = hg(x)
    loss = criterion(y[0], label, 0)
    print(loss)
    loss.backward()
    optim.step()
    optim.clear_grad()
    print(hg.down_sample[0].weight[0, 0])
