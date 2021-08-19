# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
import glob
import os
import logging
import re
import time
from pathlib import Path

import paddle


def get_optimizer(cfg, model:paddle.nn.Layer):
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


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path