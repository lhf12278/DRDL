# encoding: utf-8
"""
@author:  Kaixiong Xu
@contact: xukaixiong@stu.kust.edu.cn
"""
import torch
from torch import nn
from torch.nn import functional as F
import os.path as osp
import random
import time

from utils.ckpt import AverageMeter
from modeling.baseline import Baseline
from modeling.Baseline_IBN import Res50IBNaBNNeck

from layers.triplet_loss import CrossEntropyLabelSmooth, Cross_Entropy
from solver import WarmupMultiStepLR
working_dir = osp.abspath(osp.join(osp.dirname("__file__"), osp.pardir))

from modeling.networks import print_network, get_scheduler, cam_Classifier, \
      init_weights, Cam_Encoder, Person_Classifier

class Base_model(object):

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self._init_models()
        self._init_optimizers()

        # print('---------- Networks initialized -------------')
        # print_network(self.Content_Encoder, logger)
        # print_network(self.Person_Classifier, logger)
        # print_network(self.Cam_Encoder, logger)
        # print_network(self.Cam_Classifier, logger)
        # print('-----------------------------------------------')

    def _init_models(self):
        # ------------------------Content encoder(E1) and Camera encoder(E2)-------------------------------------
        self.Content_Encoder = Baseline(num_classes=self.cfg.DATASETS.NUM_CLASSES_S, last_stride=1, model_path=self.cfg.MODEL.PRETRAIN_PATH,
                                     neck='bnneck', neck_feat='after', model_name='resnet50', pretrain_choice='imagenet')

        # self.Content_Encoder = Res50IBNaBNNeck(class_num=self.cfg.DATASETS.NUM_CLASSES_S, pretrained=True)

        self.Cam_Encoder = Cam_Encoder(self.cfg.DATASETS.CAMS_S + self.cfg.DATASETS.CAMS_T + 1, last_stride=1)

        # ------------------------Person classifier(W1) and Camera classifier(W2)---------------------------------
        self.Person_Classifier = Person_Classifier(2048, self.cfg.DATASETS.NUM_CLASSES_S)

        self.Cam_Classifier = cam_Classifier(self.cfg.MODEL.LATENT_DIM, self.cfg.DATASETS.CAMS_S + self.cfg.DATASETS.CAMS_T + 1)

        # --------------------------initialize weights---------------------------
        init_weights(self.Cam_Classifier)

        #----------------------------criterion----------------------------
        self.xent = CrossEntropyLabelSmooth(num_classes=self.cfg.DATASETS.NUM_CLASSES_S).cuda()
        self.CrossEntropy_loss = nn.CrossEntropyLoss().cuda()
        self.Cross_Entropy = Cross_Entropy(num_classes=self.cfg.DATASETS.NUM_CLASSES_S).cuda()

        # ----------------------CUDA------------------------
        self.Content_Encoder = torch.nn.DataParallel(self.Content_Encoder).cuda()
        self.Person_Classifier = torch.nn.DataParallel(self.Person_Classifier).cuda()
        self.Cam_Encoder = torch.nn.DataParallel(self.Cam_Encoder).cuda()
        self.Cam_Classifier = torch.nn.DataParallel(self.Cam_Classifier).cuda()

    def _init_optimizers(self):
        self.Con_optimizer = make_optimizer(self.cfg, self.Content_Encoder)

        self.P_optimizer = torch.optim.Adam(self.Person_Classifier.parameters(), lr=self.cfg.MODEL.PERSON_CLASSIFIER_LR,
                                                 betas=(self.cfg.MODEL.ADAM_BEAT1, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        self.Cam_optimizer = torch.optim.Adam(self.Cam_Encoder.parameters(), lr=self.cfg.MODEL.ADAM_LR,
                                                 betas=(self.cfg.MODEL.ADAM_BEAT1, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        self.C_optimizer = torch.optim.Adam(self.Cam_Classifier.parameters(), lr=self.cfg.MODEL.C_LR,
                                                 betas=(self.cfg.MODEL.ADAM_BEAT1, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        self.schedulers = []
        self.optimizers = []

        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.cfg))

        # ----------------------scheduler-----------------#
        self.scheduler = WarmupMultiStepLR(self.Con_optimizer, (30, 55), 0.1, 1.0 / 3, 500, "linear")

    def reset_model_status(self):
        self.Content_Encoder.train()
        self.Person_Classifier.train()
        self.Cam_Encoder.train()
        self.Cam_Classifier.train()

    def train1(self, epoch, data_loader, data_loader_t, logger, print_freq=100):
        epoch = epoch
        self.reset_model_status()

        target_iter = iter(data_loader_t)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs_s, targets_s, camids_s = self._parse_data(inputs)

            try:
                inputs_target = next(target_iter)
            except:
                target_iter = iter(data_loader_t)
                inputs_target = next(target_iter)
            inputs_t, _, camids_t = self._parse_data(inputs_target)
            trans_array = camids_t.cpu().numpy()
            for idx, t in enumerate(trans_array):
                trans_array[idx] = t + self.cfg.DATASETS.CAMS_S
            camids_t = torch.from_numpy(trans_array).cuda()

            #---content feature encode
            gap_score, gap_feat_s, gmp_score, gmp_feat_s, cla_score, content_fea_s = self.Content_Encoder(inputs_s)
            _, gap_feat_s, _, gmp_feat_s, _, content_fea_t = self.Content_Encoder(inputs_t)

            #---camera feuture encode
            cam_fea_s, camfeatmap_s = self.Cam_Encoder(inputs_s)
            cam_fea_t, camfeatmap_t = self.Cam_Encoder(inputs_t)

            if (epoch < 70):# pre-learning
                # id loss
                gap_feat_id_loss = self.xent(gap_score, targets_s)
                gmp_feat_id_loss = self.xent(gmp_score, targets_s)
                softmax_loss = self.xent(cla_score, targets_s)
                CE_loss = softmax_loss

                # optimize E1 and W1
                self.P_optimizer.zero_grad()
                self.Con_optimizer.zero_grad()

                CE_loss.backward(retain_graph=True)
                gap_feat_id_loss.backward(retain_graph=True)
                gmp_feat_id_loss.backward(retain_graph=True)
                self.P_optimizer.step()
                self.Con_optimizer.step()

                # Cam_id loss
                Style_s, Cam_class_s = self.Cam_Classifier(cam_fea_s)
                Style_t, Cam_class_t = self.Cam_Classifier(cam_fea_t)
                Cam_s_loss = self.CrossEntropy_loss(Cam_class_s, camids_s)
                Cam_t_loss = self.CrossEntropy_loss(Cam_class_t, camids_t)
                Cam_class_loss = Cam_s_loss + Cam_t_loss

                # optimize E2 and W2
                if epoch % 2 == 0:
                    self.C_optimizer.zero_grad()
                    Cam_class_loss.backward(retain_graph=True)
                    self.C_optimizer.step()
                else:
                    self.Cam_optimizer.zero_grad()
                    Cam_class_loss.backward(retain_graph=True)
                    self.Cam_optimizer.step()

                total_loss = CE_loss + Cam_class_loss

                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) % print_freq == 0:
                    logger.info('Epoch:[{}/{}][{}/{}], '
                                'Time:{:.3f} ({:.3f}), '
                                'Data:{:.3f} ({:.3f}), '
                                'total_Loss:{:.3f}, ID_loss:{:.3f}, Cam_class_loss:{:.3f},'
                                .format(epoch, self.cfg.MODEL.TRAIN_EPOCH, i + 1, len(data_loader),
                                        batch_time.val, batch_time.avg,
                                        data_time.val, data_time.avg,
                                        total_loss.item(), CE_loss.item(), Cam_class_loss.item()
                                        ))

            elif (epoch >= 70 and epoch < 160):
                Style_s, Cam_class_s = self.Cam_Classifier(cam_fea_s)
                Style_t, Cam_class_t = self.Cam_Classifier(cam_fea_t)
                Cam_s_loss = self.CrossEntropy_loss(Cam_class_s, camids_s)
                Cam_t_loss = self.CrossEntropy_loss(Cam_class_t, camids_t)

                Cam_class_loss = Cam_s_loss + Cam_t_loss

                # Fix E1 and optimize W2
                _, Content_cam_s = self.Cam_Classifier(content_fea_s)
                _, Content_cam_t = self.Cam_Classifier(content_fea_t)
                Content_cam_loss = self.CrossEntropy_loss(Content_cam_s, camids_s) + self.CrossEntropy_loss(Content_cam_t, camids_t)

                Cam_loss = Cam_class_loss + Content_cam_loss

                # Fix E2 and optimize W1
                Cam_content_s, _ = self.Person_Classifier(cam_fea_s)
                Cam_content_loss1 = self.xent(Cam_content_s, targets_s)
                Cam_content_loss = self.cfg.MODEL.WEIGHT_TAO * Cam_content_loss1

                # optimize E2
                cams_use_pcls, _ = self.Person_Classifier(cam_fea_s)
                EP_loss = self.Cross_Entropy(cams_use_pcls)
                EP_loss1 = self.cfg.MODEL.WEIGHT_LAMBDA * EP_loss

                if epoch % 2 == 0:
                    # optimize W2
                    self.C_optimizer.zero_grad()
                    Cam_loss.backward(retain_graph=True)
                    self.C_optimizer.step()

                    # optimize W1
                    self.P_optimizer.zero_grad()
                    Cam_content_loss.backward(retain_graph=True)
                    self.P_optimizer.step()

                else:
                    # optimize E2
                    self.Cam_optimizer.zero_grad()
                    Cam_loss.backward(retain_graph=True)
                    EP_loss1.backward(retain_graph=True)
                    self.Cam_optimizer.step()

                # optimize E1
                last_label = torch.tensor(self.cfg.DATASETS.CAMS_S + self.cfg.DATASETS.CAMS_T).expand(
                    self.cfg.SOLVER.IMS_PER_BATCH).cuda()
                # Both source and target content features are categorized into the Cs+Ct+1 camera category
                _, source_use_camcls = self.Cam_Classifier(content_fea_s)
                _, target_use_camcls = self.Cam_Classifier(content_fea_t)
                Cam_mix_loss = self.CrossEntropy_loss(source_use_camcls, last_label) + self.CrossEntropy_loss(target_use_camcls, last_label)

                Cam_mix_loss2 = self.cfg.MODEL.WEIGHT_ALPHA * Cam_mix_loss

                # id loss
                gap_feat_id_loss = self.xent(gap_score, targets_s)
                gmp_feat_id_loss = self.xent(gmp_score, targets_s)
                softmax_loss = self.xent(cla_score, targets_s)

                CE_loss = softmax_loss + Cam_mix_loss2

                self.P_optimizer.zero_grad()
                self.Con_optimizer.zero_grad()
                CE_loss.backward(retain_graph=True)
                gap_feat_id_loss.backward(retain_graph=True)
                gmp_feat_id_loss.backward(retain_graph=True)
                self.P_optimizer.step()
                self.Con_optimizer.step()

                total_loss = Cam_class_loss + Content_cam_loss + softmax_loss + Cam_mix_loss2 +  EP_loss1

                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) % print_freq == 0:
                    logger.info('Epoch:[{}/{}][{}/{}], '
                                'Time:{:.3f} ({:.3f}), '
                                'Data:{:.3f} ({:.3f}), '
                                'total_Loss:{:.3f}, '
                                'ID_loss:{:.3f}, Cam_mix_loss:{:.3f}, '
                                'Cam_class_loss:{:.3f}, Content_cam_loss:{:.3f}, EP_loss:{:.3f}, Cam_content_loss:{:.3f}, '
                                .format(epoch, self.cfg.MODEL.TRAIN_EPOCH, i + 1, len(data_loader),
                                        batch_time.val, batch_time.avg,
                                        data_time.val, data_time.avg,
                                        total_loss.item(),
                                        softmax_loss.item(), Cam_mix_loss2.item(),
                                        Cam_class_loss.item(), Content_cam_loss.item(), EP_loss1.item(), Cam_content_loss.item(),
                                        ))
        self.scheduler.step(epoch)

    # For prid2011 and grid
    def train2(self, epoch, data_loader_s, data_loader_t, logger, print_freq=100):
        epoch = epoch
        self.reset_model_status()

        target_iter = iter(data_loader_t)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        if (epoch < 70 ):
            for i, inputs in enumerate(data_loader_s):
                data_time.update(time.time() - end)
                inputs_s, targets_s, camids_s = self._parse_data(inputs)

                try:
                    inputs_target = next(target_iter)
                except:
                    target_iter = iter(data_loader_t)
                    inputs_target = next(target_iter)
                inputs_t, _, camids_t = self._parse_data(inputs_target)
                trans_array = camids_t.cpu().numpy()
                for idx, t in enumerate(trans_array):
                    trans_array[idx] = t + self.cfg.DATASETS.CAMS_S
                camids_t = torch.from_numpy(trans_array).cuda()

                gap_score, gap_feat_s, gmp_score, gmp_feat_s, cla_score, content_fea_s = self.Content_Encoder(inputs_s)
                _, gap_feat_s, _, gmp_feat_s, _, content_fea_t = self.Content_Encoder(inputs_t)

                cam_fea_s, camfeatmap_s = self.Cam_Encoder(inputs_s)
                cam_fea_t, camfeatmap_t = self.Cam_Encoder(inputs_t)

                gap_feat_id_loss = self.xent(gap_score, targets_s)
                gmp_feat_id_loss = self.xent(gmp_score, targets_s)
                softmax_loss = self.xent(cla_score, targets_s)

                CE_loss = softmax_loss

                self.P_optimizer.zero_grad()
                self.Con_optimizer.zero_grad()
                CE_loss.backward(retain_graph=True)
                gap_feat_id_loss.backward(retain_graph=True)
                gmp_feat_id_loss.backward(retain_graph=True)
                self.P_optimizer.step()
                self.Con_optimizer.step()

                Style_s, Cam_class_s = self.Cam_Classifier(cam_fea_s)
                Style_t, Cam_class_t = self.Cam_Classifier(cam_fea_t)
                Cam_s_loss = self.CrossEntropy_loss(Cam_class_s, camids_s)
                Cam_t_loss = self.CrossEntropy_loss(Cam_class_t, camids_t)
                Cam_class_loss = Cam_s_loss + Cam_t_loss

                if epoch % 2 == 0:
                    self.C_optimizer.zero_grad()
                    Cam_class_loss.backward(retain_graph=True)
                    self.C_optimizer.step()
                else:
                    self.Cam_optimizer.zero_grad()
                    Cam_class_loss.backward(retain_graph=True)
                    self.Cam_optimizer.step()

                total_loss = CE_loss + Cam_class_loss

                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) % print_freq == 0:
                    logger.info('Epoch:[{}/{}][{}/{}], '
                                'Time:{:.3f} ({:.3f}), '
                                'Data:{:.3f} ({:.3f}), '
                                'total_Loss:{:.5f}, ID_loss:{:.5f}, Cam_class_loss:{:.5f}, '
                                .format(epoch, self.cfg.MODEL.TRAIN_EPOCH, i + 1, len(data_loader_s),
                                        batch_time.val, batch_time.avg,
                                        data_time.val, data_time.avg,
                                        total_loss.item(), CE_loss.item(), Cam_class_loss.item(),
                                        ))
        else:
            for i, input in enumerate(data_loader_t):
                source_iter = iter(data_loader_s)
                data_time.update(time.time() - end)
                inputs_t, _, camids_t = self._parse_data(input)

                try:
                    inputs_source = next(source_iter)
                except:
                    source_iter = iter(data_loader_s)
                    inputs_source = next(source_iter)
                inputs_s, targets_s, camids_s = self._parse_data(inputs_source)
                trans_array = camids_t.cpu().numpy()
                for idx, t in enumerate(trans_array):
                    trans_array[idx] = t + self.cfg.DATASETS.CAMS_S
                camids_t = torch.from_numpy(trans_array).cuda()

                gap_score, gap_feat_s, gmp_score, gmp_feat_s, cla_score, content_fea_s = self.Content_Encoder(
                    inputs_s)
                _, gap_feat_t, _, gmp_feat_t, _, content_fea_t = self.Content_Encoder(inputs_t)

                cam_fea_s, camfeatmap_s = self.Cam_Encoder(inputs_s)
                cam_fea_t, camfeatmap_t = self.Cam_Encoder(inputs_t)
                # ====================================================================================================================================#

                Style_s, Cam_class_s = self.Cam_Classifier(cam_fea_s)
                Style_t, Cam_class_t = self.Cam_Classifier(cam_fea_t)
                Cam_s_loss = self.CrossEntropy_loss(Cam_class_s, camids_s)
                Cam_t_loss = self.CrossEntropy_loss(Cam_class_t, camids_t)

                Cam_class_loss = Cam_s_loss + Cam_t_loss

                _, Content_cam_s = self.Cam_Classifier(content_fea_s)
                _, Content_cam_t = self.Cam_Classifier(content_fea_t)
                Content_cam_loss = self.CrossEntropy_loss(Content_cam_s, camids_s) + self.CrossEntropy_loss(
                    Content_cam_t, camids_t)

                Cam_loss = Cam_class_loss + Content_cam_loss

                Cam_content_s, _ = self.Person_Classifier(cam_fea_s)
                Cam_content_loss1 = self.xent(Cam_content_s, targets_s)
                Cam_content_loss = self.cfg.MODEL.WEIGHT_TAO * Cam_content_loss1

                cams_use_pcls, _ = self.Person_Classifier(cam_fea_s)
                EP_loss = self.Cross_Entropy(cams_use_pcls)
                EP_loss1 = self.cfg.MODEL.WEIGHT_LAMBDA * EP_loss

                if epoch % 2 == 0:
                    self.C_optimizer.zero_grad()
                    Cam_loss.backward(retain_graph=True)
                    self.C_optimizer.step()

                    self.P_optimizer.zero_grad()
                    Cam_content_loss.backward(retain_graph=True)
                    self.P_optimizer.step()
                else:
                    self.Cam_optimizer.zero_grad()
                    Cam_loss.backward(retain_graph=True)
                    EP_loss1.backward(retain_graph=True)
                    self.Cam_optimizer.step()

                last_label = torch.tensor(self.cfg.DATASETS.CAMS_S + self.cfg.DATASETS.CAMS_T).expand(
                    self.cfg.SOLVER.IMS_PER_BATCH).cuda()

                _, source_use_camcls = self.Cam_Classifier(content_fea_s)
                _, target_use_camcls = self.Cam_Classifier(content_fea_t)
                Cam_mix_loss = self.CrossEntropy_loss(source_use_camcls, last_label) + self.CrossEntropy_loss(
                    target_use_camcls, last_label)

                Cam_mix_loss2 = self.cfg.MODEL.WEIGHT_ALPHA * Cam_mix_loss

                gap_feat_id_loss = self.xent(gap_score, targets_s)
                gmp_feat_id_loss = self.xent(gmp_score, targets_s)
                softmax_loss = self.xent(cla_score, targets_s)

                CE_loss = softmax_loss + Cam_mix_loss2

                self.P_optimizer.zero_grad()
                self.Con_optimizer.zero_grad()
                CE_loss.backward(retain_graph=True)
                gap_feat_id_loss.backward(retain_graph=True)
                gmp_feat_id_loss.backward(retain_graph=True)
                self.P_optimizer.step()
                self.Con_optimizer.step()

                total_loss = Cam_class_loss + Content_cam_loss + softmax_loss + Cam_mix_loss2 + EP_loss1

                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) % print_freq == 0:
                    logger.info('Epoch:[{}/{}][{}/{}], '
                                'Time:{:.3f} ({:.3f}), '
                                'Data:{:.3f} ({:.3f}), '
                                'total_Loss:{:.5f}, '
                                'ID_loss:{:.5f}, Cam_mix_loss:{:.5f}, '
                                'Cam_class_loss:{:.5f}, Content_cam_loss:{:.5f}, EP_loss:{:.8f}, Cam_content_loss:{:.8f}, '
                                .format(epoch, self.cfg.MODEL.TRAIN_EPOCH, i + 1, len(data_loader_t),
                                        batch_time.val, batch_time.avg,
                                        data_time.val, data_time.avg,
                                        total_loss.item(),
                                        softmax_loss.item(), Cam_mix_loss2.item(),
                                        Cam_class_loss.item(), Content_cam_loss.item(), EP_loss1.item(),  Cam_content_loss.item(),
                                        ))
        self.scheduler.step(epoch)

    def _parse_data(self, input):
        image, pid, camids = input
        inputs = image.cuda()
        targets = pid.cuda()
        camids = camids.cuda()
        return inputs, targets, camids

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

def make_optimizer(cfg, model):
    global optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params = params + [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # if cfg.OPTIMIZER_NAME == 'SGD':
    #     optimizer = getattr(torch.optim, cfg.OPTIMIZER_NAME)(params, momentum=cfg.MOMENTUM)
    # else:
        optimizer = getattr(torch.optim, "Adam")(params)
    return optimizer
