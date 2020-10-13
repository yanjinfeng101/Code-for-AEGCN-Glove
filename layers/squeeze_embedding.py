# -*- coding: utf-8 -*-
# file: squeeze_embedding.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import torch
import torch.nn as nn
import numpy as np

class SqueezeEmbedding(nn.Module):
    """
    Squeeze sequence embedding length to the longest one in the batch#挤压序列嵌入长度为批次中最长的
    """#句子的表示，lstm只会作用到它实际长度的句子，而不是通过无用的padding字符
    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""#print(aspect_len),tensor([1,....,2,1])
        x_sort_idx = torch.sort(-x_len)[1].long()#二维数据，dim=1是按行排序，-的话应该是从大到小吧
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()#从小到大
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        """unpack: out"""
        out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)  # (sequence, lengths)
        out = out[0]  #
        """unsort"""
        out = out[x_unsort_idx]
        return out
