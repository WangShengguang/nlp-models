"""
https://github.com/Andras7/gpt2-pytorch
"""
import torch
from torch import nn


class GPT2Model(nn.Module):
    def __init__(self):
        super(GPT2Model, self).__init__()
