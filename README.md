# stacked-hourglass.paddle         

## 一、简介

百度AiStudio论文复现挑战赛，复现论文：[Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937)

## 二、复现精度

复现要求如下表所示。

|  网络结构    |  图片大小 |  精度         |
|--------------|-----------|---------------|
| hourglass-52 | 256x256   | mean@.1 0.317 |
| ~            | 384x384   | mean@.1 0.366 |

## 三、数据集

[MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#download)

MPII标注信息来自[HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

## 四、环境依赖

见[requirements.txt](requirements.txt)，推荐使用百度源（https://mirror.baidu.com/pypi/simple）进行安装。

Ubuntu：18.04

CUDA: 11.2.2

CUDNN: 8.1.0.77

## 五、快速开始

`python train.py`

## 六、代码结构与详细说明

## 七、模型信息


