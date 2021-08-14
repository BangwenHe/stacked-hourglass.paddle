import os
import cv2
import numpy as np
import argparse
import paddle
import paddle.vision.transforms as _transform

from lib.models.hourglass import Hourglass
from lib.dataset.mpii import MPIIDataset
from lib.models.loss import JointsMSELoss
from lib.core.function import train, validate


def main(num_epochs=200):
    model = Hourglass()
    # print(model)

    criterion = JointsMSELoss(use_target_weight=False)
    transform = paddle.vision.transforms.Compose([
        paddle.vision.transforms.ToTensor(),
        paddle.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MPIIDataset(root='/home/tclab/Dataset/MPII', image_set='train', is_train=True, transform=transform)
    valid_dataset = MPIIDataset(root='/home/tclab/Dataset/MPII', image_set='valid', is_train=False, transform=transform)
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=24, num_workers=8)
    valid_loader = paddle.io.DataLoader(dataset=valid_dataset, batch_size=4, num_workers=4)

    lr = 1e-3
    optim = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=lr, milestones=[150, 170], gamma=0.9)
    model.init_weight()
    # model.load_dict(paddle.load('output/best.pdparams'))
    print('loaded checkpoint!')

    best_perf = 0
    for i in range(num_epochs):
        train(train_loader, model, criterion, optim, i, 'output', print_freq=20)
        perf = validate(valid_loader, valid_dataset, model, criterion, 'output', print_freq=1)
        scheduler.step()

        paddle.save(model.state_dict(), f'output/hg-{i}.pdparams')
        if perf > best_perf:
            paddle.save(model.state_dict(), f'output/best.pdparams')
            best_perf = perf
        print('save checkpoint to output/test')


if __name__ == '__main__':
    main()