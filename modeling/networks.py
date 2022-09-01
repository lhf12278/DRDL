# encoding: utf-8
"""
@author:  Kaixiong Xu
@contact: xukaixiong@stu.kust.edu.cn
"""
import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname== 'Conv2dBlock':
        a = True
    else:
        if classname.find('Conv') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

def init_weights(net):
    net.apply(weights_init_normal)

def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def print_network(net, logger):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(net)
    logger.info('Total number of parameters: %d' % num_params)

def get_scheduler(optimizer, cfg, iterations=-1): # args.step_size, args.gamma,  args.lr_policy
    if cfg.MODEL.LR_POLICY=='step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.MODEL.LR_STEP_SIZE,
                                        gamma=cfg.MODEL.LR_GAMMA, last_epoch=iterations)
    elif cfg.MODEL.LR_POLICY=='multistep':

        step = cfg.MODEL.LR_STEP_SIZE
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[
                                                 step + step ,  # 80
                                                 step + step + step//2,  # 100
                                                 step + step + step
                                                         ],
                                             gamma=cfg.MODEL.LR_GAMMA , last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', cfg.MODEL.LR_POLICY)
    return scheduler

################################Camera encoder(E2)#####################################
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
class Cam_Encoder(nn.Module):
    in_planes = 2048
    def __init__(self, cam_num, last_stride):
        super(Cam_Encoder, self).__init__()
        self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cam_num = cam_num
        # self.classifier = nn.Linear(self.in_planes, cam_num)

    def forward(self, x):
        camfeatmap = self.base(x)
        style_fea = self.gap(camfeatmap)
        style_fea = style_fea.view(style_fea.shape[0], -1)
        return style_fea, camfeatmap


################################Person classifier(w1)#####################################
class Person_Classifier(nn.Module):
    def __init__(self, in_dim, num_class):
        super(Person_Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_class = num_class

        self.BN = nn.BatchNorm1d(self.in_dim)
        self.BN.bias.requires_grad_(False)

        self.classifier = nn.Linear(self.in_dim, self.num_class, bias=False)

        self.BN.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat_afterBN = self.BN(x)

        cls_score = self.classifier(feat_afterBN)
        return cls_score, feat_afterBN

################################Camera classifier(W2)#####################################
class cam_Classifier(nn.Module):
    def __init__(self, embed_dim, cam_class):
        super(cam_Classifier, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(5):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)

            )
            hidden_size = hidden_size // 2  # 512-32
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, cam_class)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(5):
            hidden = self.layers[i](hidden)
        style_cls_feature = hidden.squeeze(2)
        domain_clss = self.Liner(style_cls_feature)
        return style_cls_feature, domain_clss  # [batch,15]

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)






