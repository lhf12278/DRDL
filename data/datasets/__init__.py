# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

modified by Kaixiong Xu
"""

from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .prid import PRID
from .grid import GRID
from .veri import VeRi
from .dataset_loader import ImageDataset, ImageDataset_val

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
    'grid': GRID,
    'prid': PRID,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
