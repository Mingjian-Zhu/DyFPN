import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS
from .fpn import FPN
import pdb as ipdb


class Attention_SEblock(nn.Module):
    def __init__(self, channels, reduction, temperature):
        super(Attention_SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2)
        self.fc2.bias.data[0] = 0.1 
        self.fc2.bias.data[1] = 2
        self.temperature = temperature
        self.channels = channels
    def forward(self, x):
        x = self.avg_pool(x).view(-1, self.channels)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.gumbel_softmax(x, tau=1, hard=True)
        return x

@NECKS.register_module()
class DyFPN_B_CNNGate(FPN):
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
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(DyFPN_B_CNNGate, self).__init__(in_channels, out_channels, num_outs)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.gate_0 = Attention_SEblock(channels=in_channels[0],
                                                 reduction=4, temperature=1)
        self.gate_1 = Attention_SEblock(channels=in_channels[1],
                                                 reduction=4, temperature=1)
        self.gate_2 = Attention_SEblock(channels=in_channels[2],
                                                 reduction=4, temperature=1)
        self.gate_3 = Attention_SEblock(channels=in_channels[3],
                                                 reduction=4, temperature=1)

        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.skip_lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv_11 = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            l_conv_33 = ConvModule(
                in_channels[i],
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            l_conv_33_dilation_2 = ConvModule(
                in_channels[i],
                out_channels,
                3,
                padding=2,
                dilation=2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            l_conv_33_dilation_3 = ConvModule(
                in_channels[i],
                out_channels,
                3,
                padding=3,
                dilation=3,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            l_conv_55 = ConvModule(
                in_channels[i],
                out_channels,
                5,
                padding=2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            l_conv_55_dilation_2 = ConvModule(
                in_channels[i],
                out_channels,
                5,
                padding=4,
                dilation=2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            l_conv_55_dilation_3 = ConvModule(
                in_channels[i],
                out_channels,
                5,
                padding=6,
                dilation=3,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            skip_l_conv_11 = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv_11)
            self.lateral_convs.append(l_conv_33)
            self.lateral_convs.append(l_conv_33_dilation_2)
            self.lateral_convs.append(l_conv_33_dilation_3)
            self.lateral_convs.append(l_conv_55)
            self.lateral_convs.append(l_conv_55_dilation_2)
            self.lateral_convs.append(l_conv_55_dilation_3)
            self.skip_lateral_convs.append(skip_l_conv_11)

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        laterals_topdown0_decision = self.gate_0(inputs[0])
        laterals_topdown1_decision = self.gate_1(inputs[1])
        laterals_topdown2_decision = self.gate_2(inputs[2])
        laterals_topdown3_decision = self.gate_3(inputs[3])

        laterals_0 = []
        laterals_1 = []
        laterals_2 = []
        laterals_3 = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if (i>=0 and i<=6):
                if laterals_topdown0_decision[:,1] == 1:
                    laterals_0.append(lateral_conv(inputs[0]))
            elif (i>=7 and i<=13):
                if laterals_topdown1_decision[:,1] == 1:
                    laterals_1.append(lateral_conv(inputs[1]))
            elif (i>=14 and i<=20):
                if laterals_topdown2_decision[:,1] == 1:
                    laterals_2.append(lateral_conv(inputs[2]))
            elif (i>=21 and i<=27):
                if laterals_topdown3_decision[:,1] == 1:
                    laterals_3.append(lateral_conv(inputs[3]))

        skip_lateral_conv_collection = []
        for i, skip_lateral_conv in enumerate(self.skip_lateral_convs):
            skip_lateral_conv_collection.append(skip_lateral_conv(inputs[i]))
        laterals_sum_0 = skip_lateral_conv_collection[0]
        if laterals_topdown0_decision[:,1] == 1:
            laterals_sum_0 = laterals_sum_0 + (laterals_0[0] + laterals_0[1] + laterals_0[2] + laterals_0[3] + laterals_0[4] +
                              laterals_0[5] + laterals_0[6])

        laterals_sum_1 = skip_lateral_conv_collection[1]
        if laterals_topdown1_decision[:,1] == 1:
            laterals_sum_1 = laterals_sum_1 + (laterals_1[0] + laterals_1[1] + laterals_1[2] + laterals_1[3] + laterals_1[4] +
                              laterals_1[5] + laterals_1[6])

        laterals_sum_2 = skip_lateral_conv_collection[2]
        if laterals_topdown2_decision[:, 1] == 1:
            laterals_sum_2 = laterals_sum_2 + (laterals_2[0] + laterals_2[1] + laterals_2[2] + laterals_2[3] + laterals_2[4] +
                             laterals_2[5] + laterals_2[6])

        laterals_sum_3 = skip_lateral_conv_collection[3]
        if laterals_topdown3_decision[:, 1] == 1:
            laterals_sum_3 = laterals_sum_3 + (laterals_3[0] + laterals_3[1] + laterals_3[2] + laterals_3[3] + laterals_3[4] +
                             laterals_3[5] + laterals_3[6])

        laterals = []
        laterals.append(laterals_sum_0)
        laterals.append(laterals_sum_1)
        laterals.append(laterals_sum_2)
        laterals.append(laterals_sum_3)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


