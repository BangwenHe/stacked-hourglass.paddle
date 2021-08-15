# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import os
import logging
import time

import paddle

from lib.config import Config


def get_optimizer(cfg: Config, model:paddle.nn.Layer):
    if cfg.optimizer == 'adam':
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=cfg.lr
        )
    elif cfg.optimizer == 'rmsprop':
        # https://ruder.io/optimizing-gradient-descent/index.html#rmsprop
        optimizer = paddle.optimizer.RMSProp(
            parameters=model.parameters(),
            learning_rate=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            rho=cfg.alpha,
            epsilon=cfg.epsilon,
        )
    else:
        raise NotImplementedError(f'{cfg.optimizer} is not implemented! Please try Adam or RMSprop')

    return optimizer


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pdparams'):
    paddle.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        paddle.save(states['state_dict'], os.path.join(output_dir, 'best.pdparams'))


def model_parameters(model, inputs=(1, 3, 256, 256)):
    import paddleslim
    flops = paddleslim.flops(model, inputs)
    params = paddle.summary(model, inputs)
    print(params)

    cnt = 0
    while flops > 1024:
        flops /= 1024
        cnt += 1
    suffix = ['flops', 'K flops', 'M flops', 'G flops', 'T flops']
    print(f'model float operating per second: {flops:.2f}{suffix[cnt]}')
