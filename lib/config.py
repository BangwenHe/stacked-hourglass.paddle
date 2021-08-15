import os
import os.path as osp
import re
import sys
import math
import numpy as np


class Config:
    ## dataset
    dataset_name = 'MPII'
    dataset_dir = '/home/tclab/Dataset/MPII'
    num_joints = 16
    data_format = 'jpg'

    input_img_shape = (256, 256)
    output_hm_shape = (64, 64)
    sigma = 2

    scale_factor = 0.25
    flip_factor = 0.5
    rotation_factor = 0.5

    ## model
    model_name = 'hourglass'
    num_stacked_modules = 4

    ## training config
    train_batch_size = 26
    train_workers = 4
    start_epoch = 0
    end_epoch = 200
    train_print_freq = 100
    lr = 1e-3
    optimizer = 'adam'
    decay_epoch = [150, 170]
    decay_gamma = 0.9

    ### RMSProp parameters
    learning_rate_decay = 0
    momentum = 0
    weight_decay = 0
    alpha = 0.99
    epsilon = 1e-8

    ## validing config
    valid_batch_size = 4
    valid_workers = 4
    valid_print_freq = 100

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output', model_name, dataset_name)
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')

    ## debug
    save_batch_images_gt = True
    save_batch_images_pred = True
    save_heatmaps_gt = True
    save_heatmaps_pred = True

    ## others
    gpu_ids = '3'
    num_gpus = 1
    continue_train = True
    best_model_path = os.path.join(model_dir, 'best.pdparams')
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pdparams')

    def set_args(self, gpu_ids=None, continue_train=None):
        if gpu_ids is not None:
            self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))

        if continue_train is not None:
            self.continue_train = continue_train

        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

    def prepare_output_directories(self):
        dirs = [self.output_dir, self.model_dir, self.vis_dir, self.log_dir, self.result_dir]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def print_configurations(self):
        params = [i for i in dir(self) if re.match(r'^__.*__$', i) is None and not hasattr(self.__getattribute__(i), '__call__')]
        for p in params:
            print(f'{p}: {self.__getattribute__(p)}')


cfg = Config()

if __name__ == '__main__':
    cfg.print_configurations()