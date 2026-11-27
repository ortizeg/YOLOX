#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)


def _link_or_copy(src, dst):
    """Make ``dst`` the same file as ``src``: a hardlink when the filesystem allows it, else a copy. ``dst`` is
    replaced atomically either way (built under a temporary name, then ``os.replace``)."""
    tmp = dst + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    try:
        os.link(src, tmp)
    except OSError:  # no hardlinks here (some network / FUSE filesystems): fall back to a copy
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _save_atomic(state, filename):
    """torch.save to a temporary name, then ``os.replace``: the target gets a new inode, so hardlinks made to the
    previous version (e.g. ``last_epoch_ckpt.pth``) keep their content, and a crash mid-write leaves the old file."""
    tmp = filename + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, filename)


def save_checkpoint_lean(state, save_dir, full=(), lean=()):
    """``lean_ckpt`` writer. ``full`` names get the whole ``state`` (optimizer included: resume points); ``lean``
    names get ``state`` without its ``"optimizer"`` entry (weights for eval / export / fine-tuning; ``"best"`` is
    ``best_ckpt.pth``). Each group is serialised once: the first name is written, the others are hardlinks to it."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for names, st in ((list(full), state), (list(lean), {k: v for k, v in state.items() if k != "optimizer"})):
        if not names:
            continue
        first = os.path.join(save_dir, names[0] + "_ckpt.pth")
        _save_atomic(st, first)
        for name in names[1:]:
            _link_or_copy(first, os.path.join(save_dir, name + "_ckpt.pth"))
