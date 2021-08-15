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
from lib.utils.utils import get_optimizer, save_checkpoint, model_parameters


def main():
    cfg.set_args()
    cfg.prepare_output_directories()
    cfg.print_configurations()

    model = Hourglass(num_modules=cfg.num_stacked_modules)

    criterion = JointsMSELoss(use_target_weight=False)
    transform = _transform.Compose([
        _transform.ToTensor(),
        _transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MPIIDataset(root=cfg.dataset_dir, image_set='train', is_train=True, transform=transform)
    valid_dataset = MPIIDataset(root=cfg.dataset_dir, image_set='valid', is_train=False, transform=transform)
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=cfg.train_batch_size,
                                        num_workers=cfg.train_workers)
    valid_loader = paddle.io.DataLoader(dataset=valid_dataset, batch_size=cfg.valid_batch_size,
                                        num_workers=cfg.valid_workers)

    lr = cfg.lr
    optim = get_optimizer(cfg, model)
    scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=lr, milestones=cfg.decay_epoch, gamma=cfg.decay_gamma)
    model.init_weight()
    if cfg.continue_train and os.path.exists(cfg.checkpoint_path):
        ckpt = paddle.load(cfg.checkpoint_path)
        cfg.start_epoch = ckpt['epoch']
        model.load_dict(ckpt['state_dict'])
        print(f'loaded model state dict from {cfg.checkpoint_path}')

    model_parameters(model)

    best_perf = 0
    for i in range(cfg.start_epoch, cfg.end_epoch):
        train(train_loader, model, criterion, optim, i, cfg.vis_dir, cfg.train_print_freq)
        perf = validate(valid_loader, valid_dataset, model, criterion, cfg.vis_dir, cfg.valid_print_freq)
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
