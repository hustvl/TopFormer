# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *


@HEADS.register_module()
class SimpleHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, is_dw=False, **kwargs):
        super(SimpleHead, self).__init__(input_transform='multiple_select', **kwargs)

        embedding_dim = self.channels

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            groups=embedding_dim if is_dw else 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
    
    def agg_res(self, preds):
        outs = preds[0]
        for pred in preds[1:]:
            pred = resize(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
            outs += pred
        return outs

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        x = self.agg_res(xx)
        _c = self.linear_fuse(x)
        x = self.cls_seg(_c)
        return x