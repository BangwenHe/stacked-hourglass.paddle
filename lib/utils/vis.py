# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


import cv2
from lib.core.inference import get_max_preds

import paddle
import warnings
import math
import numpy as np
from typing import Union, Optional, List, Tuple, Text, BinaryIO


@paddle.no_grad()
def make_grid(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
              nrow: int = 8,
              padding: int = 2,
              normalize: bool = False,
              value_range: Optional[Tuple[int, int]] = None,
              scale_each: bool = False,
              pad_value: int = 0,
              **kwargs) -> paddle.Tensor:
    if not (isinstance(tensor, paddle.Tensor) or
            (isinstance(tensor, list) and all(
                isinstance(t, paddle.Tensor) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clip(min=low, max=high)
            img = img - low
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] +
                                                        padding)
    num_channels = tensor.shape[1]
    grid = paddle.full((num_channels, height * ymaps + padding,
                        width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width] = tensor[k]
            k = k + 1
    return grid


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/pytorch_project_convertor/API_docs/vision/torchvision.utils.save_image.md
    grid = make_grid(batch_image, nrow, padding, True)
    ndarr = paddle.clip(grid * 255 + 0.5, 0, 255).transpose(
        [1, 2, 0]).cast("uint8").numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.shape[2] + padding)
    width = int(batch_image.shape[3] + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1

    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image = (batch_image - min) / (max - min + 1e-5)

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    grid_image = np.zeros((batch_size * heatmap_height,
                           (num_joints + 1) * heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = paddle.clip(batch_image[i] * 255 + 0.5, 0, 255).transpose([1, 2, 0]).cast("uint8").numpy()
        heatmaps = paddle.clip(batch_heatmaps[i] * 255 + 0.5, 0, 255).transpose([1, 2, 0]).cast("uint8").numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[:, :, j]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, grid_image)


def save_debug_images(input, meta, target, joints_pred, output, prefix):
    from lib.config import cfg
    if cfg.save_batch_images_gt:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if cfg.save_batch_images_pred:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if cfg.save_heatmaps_gt:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if cfg.save_heatmaps_pred:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
