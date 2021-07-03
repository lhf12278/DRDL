import torch
import os.path as osp
from modeling.baseline import Baseline

working_dir = osp.abspath(osp.join(osp.dirname("__file__"), osp.pardir))


class Base_model(object):

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self._init_models()

    def _init_models(self):
        self.Content_Encoder = Baseline(num_classes=self.cfg.DATASETS.NUM_CLASSES_S, last_stride=1, model_path=self.cfg.MODEL.PRETRAIN_PATH,
                                     neck='bnneck', neck_feat='after', model_name='resnet50', pretrain_choice='imagenet')

        self.Content_Encoder = torch.nn.DataParallel(self.Content_Encoder).cuda()
