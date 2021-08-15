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
from lib.config import cfg
from lib.utils.utils import get_optimizer, save_checkpoint


def main():
    model = Hourglass()
    cfg.set_args(gpu_ids='3')

    criterion = JointsMSELoss(use_target_weight=False)
    transform = _transform.Compose([
        _transform.ToTensor(),
        _transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MPIIDataset(root=cfg.dataset_dir, image_set='train', is_train=True, transform=transform)
    valid_dataset = MPIIDataset(root=cfg.dataset_dir, image_set='valid', is_train=False, transform=transform)
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=cfg.train_batch_size,
                                        num_workers=cfg.train_workers)
    valid_loader = paddle.io.DataLoader(dataset=valid_dataset, batch_size=cfg.test_batch_size,
                                        num_workers=cfg.test_workers)

    lr = cfg.lr
    optim = get_optimizer(cfg, model)
    scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=lr, milestones=cfg.decay_epoch, gamma=cfg.decay_gamma)
    model.init_weight()
    # model.load_dict(paddle.load('output/best.pdparams'))
    print('loaded checkpoint!')

    best_perf = 0
    for i in range(cfg.start_epoch, cfg.end_epoch):
        train(train_loader, model, criterion, optim, i, 'output', print_freq=20)
        perf = validate(valid_loader, valid_dataset, model, criterion, 'output', print_freq=1)
        scheduler.step()

        ckpt = {
            'state_dict': model.state_dict(),
            'epoch': i,
            'optimizer': optim.state_dict()
        }
        save_checkpoint(ckpt, perf > best_perf, cfg.model_dir)
        if perf > best_perf: best_perf = perf
        print(f'save checkpoint to {cfg.model_dir}')


if __name__ == '__main__':
    main()
