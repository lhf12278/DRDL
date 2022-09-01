# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com

modified by Kaixiong Xu
"""
import torch


def train_collate_fn(batch):
    imgs, pids, camids, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids
