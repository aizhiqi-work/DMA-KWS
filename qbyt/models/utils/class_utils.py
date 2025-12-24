#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
import torch
from torch.nn import BatchNorm1d, LayerNorm
from models.attention import (MultiHeadedAttention,
                                         MultiHeadedCrossAttention,
                                         RelPositionMultiHeadedAttention,
                                         RopeMultiHeadedAttention,
                                         ShawRelPositionMultiHeadedAttention)
from models.embedding import (
    LearnablePositionalEncoding, NoPositionalEncoding, PositionalEncoding,
    RelPositionalEncoding, RopePositionalEncoding)
from models.norm import RMSNorm
from models.positionwise_feed_forward import (
    GatedVariantsMLP, MoEFFNLayer, PositionwiseFeedForward)
from models.subsampling import (
    Conv1dSubsampling2, Conv2dSubsampling4, Conv2dSubsampling6,
    Conv2dSubsampling8, EmbedinigNoSubsampling, LinearNoSubsampling,
    StackNFramesSubsampling)
from models.swish import Swish

WENET_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

WENET_RNN_CLASSES = {
    "rnn": torch.nn.RNN,
    "lstm": torch.nn.LSTM,
    "gru": torch.nn.GRU,
}

WENET_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
}

WENET_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
    'rope_pos': RopePositionalEncoding,
}

WENET_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
    "crossattn": MultiHeadedCrossAttention,
    'shaw_rel_selfattn': ShawRelPositionMultiHeadedAttention,
    'rope_abs_selfattn': RopeMultiHeadedAttention,
}

WENET_MLP_CLASSES = {
    'position_wise_feed_forward': PositionwiseFeedForward,
    'moe': MoEFFNLayer,
    'gated': GatedVariantsMLP
}

WENET_NORM_CLASSES = {
    'layer_norm': LayerNorm,
    'batch_norm': BatchNorm1d,
    'rms_norm': RMSNorm
}
