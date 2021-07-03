# -*- coding: utf-8 -*-
# @Author   : xukaixiong
# @Time     : 2020/5/16 9:58
# @direction: test

from __future__ import print_function, absolute_import
import os.path as osp
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath) # server will occur an error "no module named cfg..." if delete this
from config import cfg
from data import make_data_loader, make_data_loader_target
from engine.trainer import create_supervised_evaluator
from utils.reid_metric import R1_mAP, R1_mAP_reranking
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from modeling.model import Base_model
from utils.ckpt import load_checkpoint, save_checkpoint
from utils.logger import setup_logger
working_dir = osp.abspath(osp.join(osp.dirname("__file__"), osp.pardir)) # reid-strong-baseline master


def main():
    logger = setup_logger("duke2market", cfg.OUTPUT_DIR, 0, '0214test')
    # logger.info(cfg)
    # args = Arguments().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    # ----load dataset------ #
    train_loader_s, _, _, num_classes = make_data_loader(cfg)
    train_loader_t, val_loader, num_query, _ = make_data_loader_target(cfg)

    cfg.DATASETS.NUM_CLASSES_S = num_classes
    my_model = Base_model(cfg, logger) # --------------
    # Evaluator
    if cfg.TEST.RE_RANKING == 'no':
        evaluator = create_supervised_evaluator(my_model.Content_Encoder,
                                            metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm='yes')}, device='cuda')
    else:
        evaluator = create_supervised_evaluator(my_model.Content_Encoder,
                                            metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device='cuda')

    # ---------------------test------------------------#
    model_checkpoint = load_checkpoint(osp.join(working_dir, 'logs/0214_duke2market/duke2market-new.pth.tar'))
    my_model.Content_Encoder.module.load_state_dict(model_checkpoint['Content_Encoder'])
    logger.info("=> Training on {} and Testing on {}".format(cfg.DATASETS.NAMES, cfg.DATASETS.TNAMES))
    print("=> start testing. Please wait...")
    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info("Validation Result using best model")
    logger.info("mAP: {:.1%}".format(mAP))
    for i in [1, 5, 10, 20, 30, 50]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(i, cmc[i - 1]))

    logger.info("finished!")

if __name__ == '__main__':
    main()
