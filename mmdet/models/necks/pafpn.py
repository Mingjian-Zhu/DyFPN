import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.core import auto_fp16
from ..builder import NECKS
from .fpn import FPN
import pdb as ipdb

@NECKS.register_module()
class PAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(PAFPN,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, add_extra_convs, extra_convs_on_inputs,
                             relu_before_extra_convs, no_norm_on_lateral,
                             conv_cfg, norm_cfg, act_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.downsample_convs_1 = nn.ModuleList()
        self.downsample_convs_2 = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        self.lateral_convs_1 = nn.ModuleList()
        self.lateral_convs_2 = nn.ModuleList()
        self.fpn_convs_1 = nn.ModuleList()
        self.fpn_convs_2 = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            d_conv_1 = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            d_conv_2 = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.downsample_convs_1.append(d_conv_1)
            self.downsample_convs_2.append(d_conv_2)
            self.pafpn_convs.append(pafpn_conv)
        self.convs_between_blocks_0 = nn.ModuleList()
        self.convs_between_blocks_1 = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv_1 = ConvModule(
                out_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv_1 = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            l_conv_2 = ConvModule(
                out_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv_2 = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs_1.append(l_conv_1)
            self.fpn_convs_1.append(fpn_conv_1)
            self.lateral_convs_2.append(l_conv_2)
            self.fpn_convs_2.append(fpn_conv_2)


    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        ##########################################
        # Part 1
        ##########################################
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        inter_outs_0 = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        #self.fpn_conv inherient from fpn

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs_0[i + 1] += self.downsample_convs[i](inter_outs_0[i])

        ##########################################
        # Part 2
        ##########################################
        # build laterals
        laterals_1 = [
            lateral_conv_1(inter_outs_0[i + self.start_level])
            for i, lateral_conv_1 in enumerate(self.lateral_convs_1)
        ]

        # build top-down path
        used_backbone_levels_1 = len(laterals_1)
        for i in range(used_backbone_levels_1 - 1, 0, -1):
            prev_shape = laterals_1[i - 1].shape[2:]
            laterals_1[i - 1] += F.interpolate(
                laterals_1[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        inter_outs_1 = [
            self.fpn_convs_1[i](laterals_1[i]) for i in range(used_backbone_levels_1)
        ]
        # self.fpn_conv inherient from fpn

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels_1 - 1):
            inter_outs_1[i + 1] += self.downsample_convs_1[i](inter_outs_1[i])

        ##########################################
        # Part 3
        ##########################################
        # build laterals
        laterals_2 = [
            lateral_conv_2(inter_outs_1[i + self.start_level])
            for i, lateral_conv_2 in enumerate(self.lateral_convs_2)
        ]

        # build top-down path
        used_backbone_levels_2 = len(laterals_2)
        for i in range(used_backbone_levels_2 - 1, 0, -1):
            prev_shape = laterals_2[i - 1].shape[2:]
            laterals_2[i - 1] += F.interpolate(
                laterals_2[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        inter_outs_2 = [
            self.fpn_convs_2[i](laterals_2[i]) for i in range(used_backbone_levels_2)
        ]
        # self.fpn_conv inherient from fpn

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels_2 - 1):
            inter_outs_2[i + 1] += self.downsample_convs_2[i](inter_outs_2[i])


        outs = []
        outs.append(inter_outs_2[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs_2[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
