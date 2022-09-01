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
working_dir = osp.abspath(osp.join(osp.dirname("__file__"), osp.pardir))


def main():
    logger = setup_logger("Duke2Market", cfg.OUTPUT_DIR, 0, 'Duke2Market.txt')
    logger.info(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    # ----load dataset------ #
    train_loader_s, _, _, num_classes = make_data_loader(cfg)
    train_loader_t, val_loader, num_query, _ = make_data_loader_target(cfg)
    # modify args, check BaseModel, dict
    cfg.DATASETS.NUM_CLASSES_S = num_classes
    my_model = Base_model(cfg, logger) # --------------
    # Evaluator
    if cfg.TEST.RE_RANKING == 'no':
        evaluator = create_supervised_evaluator(my_model.Content_Encoder,
                                            metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm='yes')}, device='cuda')
    else:
        evaluator = create_supervised_evaluator(my_model.Content_Encoder,
                                            metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device='cuda')

    # Summary_writer
    writer = SummaryWriter()
    # Start training
    start_epoch = best_top1 = 0
    for epoch in range(start_epoch, cfg.MODEL.TRAIN_EPOCH):
        my_model.train1(epoch, train_loader_s, train_loader_t, logger)
        # my_model.train2(epoch, train_loader_s, train_loader_t, logger)  # For prid2011 and grid
        if (epoch+1) % cfg.MODEL.EVAL_PERIOD == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            is_best = cmc[0] > best_top1
            best_top1 = max(cmc[0], best_top1)
            save_checkpoint({
                'Content_Encoder': my_model.Content_Encoder.module.state_dict(),
                'Cam_Encoder': my_model.Cam_Encoder.module.state_dict(),
                'Cam_Classifier': my_model.Cam_Classifier.module.state_dict(),
                'Con_optimizer': my_model.Con_optimizer.state_dict(),
                'Cam_optimizer': my_model.Cam_optimizer.state_dict(),
                'C_optimizer': my_model.C_optimizer.state_dict(),
                'epoch':  epoch + 1,
                'best_top1': best_top1,
            }, is_best, fpath=osp.join(working_dir, 'logs/Duke2Market', 'D2M-checkpoint.pth.tar'))
            logger.info('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                        format(epoch, cmc[0], best_top1, ' *' if is_best else ''))
        my_model.update_learning_rate()
        if(epoch+1 == 70):
            save_checkpoint(
                {
                    'Content_Encoder': my_model.Content_Encoder.module.state_dict(),
                    'Cam_Encoder': my_model.Cam_Encoder.module.state_dict(),
                    'Cam_Classifier': my_model.Cam_Classifier.module.state_dict(),
                    'Person_Classifier': my_model.Person_Classifier.module.state_dict(),
                    'Con_optimizer': my_model.Con_optimizer.state_dict(),
                    'Cam_optimizer': my_model.Cam_optimizer.state_dict(),
                    'C_optimizer': my_model.C_optimizer.state_dict(),
                    'P_optimizer': my_model.P_optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best=False, fpath=osp.join(working_dir, 'logs/Duke2Market',
                                                 'D2M_{}-checkpoint.pth.tar'.format(epoch + 1))
            )

    writer.close()

    logger.info("finished!")

if __name__ == '__main__':
    main()


