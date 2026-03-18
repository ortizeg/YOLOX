#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch

__all__ = ["meshgrid"]


def meshgrid(*tensors):
    return torch.meshgrid(*tensors, indexing="ij")
