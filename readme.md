# DyFPN

## Introduction
Dynamic Feature Pyramid Networks for Object Detection. [arXiv](https://arxiv.or
g/abs/2012.00779)

By Mingjian Zhu, Kai Han, Changbin Yu, Yunhe Wang


This is the implementation of DyFPN. Basically, we follow the setting of testing 
a model in MMDetection. Please refer to [MMDetection](https://github.com/open-mmlab/mmdetection) for installation and dataset preparation.


## Test
1. Download the pre-trained model in [Onedrive](https://westlakeu-my.sharepoi
nt.com/:u:/g/personal/zhumingjian_westlake_edu_cn/ESmbMPHJ3SxDk6Hfkw
HDrqwBiVZN-fDPNSnNSj4Tq7VGOA?e=CQ8U97) and put it in the *checkpoint* folder.

2. Test the model with the following command.
```
python tools/test.py configs/dyfpn/faster_rcnn_r50_dyfpn_1x_coco.py checkpoi
nts/DyFPN_B_CNNGate.pth --eval bbox
```

## Citation
```
@article{zhu2020dynamic,
  title={Dynamic Feature Pyramid Networks for Object Detection},
  author={Zhu, Mingjian and Han, Kai and Yu, Changbin and Wang, Yunhe},
  journal={arXiv preprint arXiv:2012.00779},
  year={2020}

```