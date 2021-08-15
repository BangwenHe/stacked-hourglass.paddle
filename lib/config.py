import os
import os.path as osp
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
    train_batch_size = 24
    train_workers = 4
    start_epoch = 0
    end_epoch = 200
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

    ## testing config
    test_batch_size = 4
    test_workers = 4

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output', model_name, dataset_name)
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')

    ## others
    gpu_ids = '3'
    num_gpus = 1
    continue_train = False

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))


cfg = Config()