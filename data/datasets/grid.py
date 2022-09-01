# -*- coding: utf-8 -*-
# @Author   : Kaixiong Xu
# @contact: xukaixiong@stu.kust.edu.cn
"""
Our partition protocol:
    125 image pairs of pedestrians and 400 interference images are randomly sampled for training.
    One image is randomly selected from each remaining pair of pedestrians as one query image,
    and there are 125 query images in total. The rest 125 images and 375 interference images are used as the gallery set.

    training set: 650 images with 125 identities
    query set: 125 images with 125 identities
    gallery set : 500 images with 126 identities
"""
import numpy as np
from PIL import Image
import copy
import os
import copy
from prettytable import PrettyTable
from easydict import EasyDict
import random
# import albumentations as A
from scipy.io import loadmat
import cv2
import glob
import os.path as osp

import errno
import json
import os

import os.path as osp


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))



def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


class PersonReIDSamples:

    def _relabels(self, samples, label_index):
        '''
        reorder labels
        map labels [1, 3, 5, 7] to [0,1,2,3]
        '''
        ids = []
        for sample in samples:
            ids.append(sample[label_index])
        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()
        # reorder
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples

    def _relabels_c(self, samples, label_index,dataset='market',s_t='market_duke',is_target=False):
        '''
        reorder labels
        map labels [1, 3, 5, 7] to [0,1,2,3]
        '''
        # reorder
        if s_t=='market_duke' or s_t=='duke_market':
            if dataset=='market' and is_target==False:
                for sample in samples:
                    if (sample[2]== 1 or sample[2]==4 or sample[2]==5):
                        sample[label_index][0][0] = sample[1]+751
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 6):
                        sample[label_index][0][1] = sample[1]+751
                        sample[label_index][0][0] = sample[1]

                    if (sample[2]==2 or sample[2]==4 or sample[2]==5 ):
                        sample[label_index][1][0] = sample[1] + 751
                        sample[label_index][1][1] = sample[1]
                    elif(sample[2]==3 or sample[2]==1 or sample[2]==6):
                        sample[label_index][1][1] = sample[1] + 751
                        sample[label_index][1][0] = sample[1]

                    if (sample[2] == 1 or sample[2] == 5 or sample[2] == 6):
                        sample[label_index][2][0] = sample[1] + 751
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 2 or sample[2] == 3):
                        sample[label_index][2][1] = sample[1] + 751
                        sample[label_index][2][0] = sample[1]

                    if (sample[2]==2 or sample[2]== 5 or sample[2]==1):
                        sample[label_index][3][0] = sample[1] + 751
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 6 or sample[2] == 3):
                        sample[label_index][3][1] = sample[1] + 751
                        sample[label_index][3][0] = sample[1]
                return samples
            elif dataset == 'market' and is_target:
                for sample in samples:
                    if (sample[2]== 1 or sample[2]==4 or sample[2]==5):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif(sample[2]== 2 or sample[2]==3 or sample[2]==6):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if (sample[2]==2 or sample[2]==4 or sample[2]==5 ):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif(sample[2]==3 or sample[2]==1 or sample[2]==6 ):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if (sample[2]==1 or sample[2]==5 or sample[2]==6):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif(sample[2]==4 or sample[2]==2 or sample[2]==3):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if (sample[2]==2 or sample[2]== 5 or sample[2]==1):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif(sample[2]==4 or sample[2]==6 or sample[2]==3):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1

            elif dataset=='duke' and is_target:
                for sample in samples:
                    if(sample[2]==5 or sample[2]==7 or sample[2]==6 or sample[2]==4):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif (sample[2] == 1 or sample[2] == 2 or sample[2] == 3 or sample[2] == 8):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if(sample[2]==5 or sample[2]==2 or sample[2]==3 or sample[2]==8):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif (sample[2] == 7 or sample[2] == 6 or sample[2] == 4 or sample[2] == 1):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if (sample[2]==7 or sample[2]==3 or sample[2]==8 or sample[2]==1):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif (sample[2] == 5 or sample[2] == 4 or sample[2] == 6 or sample[2] == 2):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if (sample[2]==6 or sample[2]==3 or sample[2]==5 or sample[2]==1):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif (sample[2] == 4 or sample[2] == 8 or sample[2] == 2 or sample[2] == 7):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1

            elif dataset == 'duke' and is_target==False:
                for sample in samples:
                    if(sample[2]==5 or sample[2]==7 or sample[2]==6 or sample[2]==4):
                        sample[label_index][0][0] = sample[1]+702
                        sample[label_index][0][1] = sample[1]
                    elif(sample[2]==1 or sample[2]==2 or sample[2]==3 or sample[2]==8):
                        sample[label_index][0][1] = sample[1]+702
                        sample[label_index][0][0] = sample[1]

                    if(sample[2]==5 or sample[2]==2 or sample[2]==3 or sample[2]==8):
                        sample[label_index][1][0] = sample[1] + 702
                        sample[label_index][1][1] = sample[1]
                    elif(sample[2]==7 or sample[2]==6 or sample[2]==4 or sample[2]==1):
                        sample[label_index][1][1] = sample[1] + 702
                        sample[label_index][1][0] = sample[1]

                    if (sample[2]==7 or sample[2]==3 or sample[2]==8 or sample[2]==1):
                        sample[label_index][2][0] = sample[1] + 702
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2]==5 or sample[2]==4 or sample[2]==6 or sample[2]==2):
                        sample[label_index][2][1] = sample[1] + 702
                        sample[label_index][2][0] = sample[1]

                    if (sample[2]==6 or sample[2]==3 or sample[2]==5 or sample[2]==1):
                        sample[label_index][3][0] = sample[1] + 702
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2]==4 or sample[2]==8 or sample[2]==2 or sample[2]==7):
                        sample[label_index][3][1] = sample[1] + 702
                        sample[label_index][3][0] = sample[1]
        elif s_t=='market_msmt' or s_t=='msmt_market':
            if dataset=='market' and is_target==False:
                for sample in samples:
                    if (sample[2]== 1 or sample[2]==4 or sample[2]==5):
                        sample[label_index][0][0] = sample[1]+751
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 6):
                        sample[label_index][0][1] = sample[1]+751
                        sample[label_index][0][0] = sample[1]

                    if (sample[2]==2 or sample[2]==4 or sample[2]==5 ):
                        sample[label_index][1][0] = sample[1] + 751
                        sample[label_index][1][1] = sample[1]
                    elif(sample[2]==3 or sample[2]==1 or sample[2]==6):
                        sample[label_index][1][1] = sample[1] + 751
                        sample[label_index][1][0] = sample[1]

                    if (sample[2] == 1 or sample[2] == 5 or sample[2] == 6):
                        sample[label_index][2][0] = sample[1] + 751
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 2 or sample[2] == 3):
                        sample[label_index][2][1] = sample[1] + 751
                        sample[label_index][2][0] = sample[1]

                    if (sample[2]==2 or sample[2]== 5 or sample[2]==1):
                        sample[label_index][3][0] = sample[1] + 751
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 6 or sample[2] == 3):
                        sample[label_index][3][1] = sample[1] + 751
                        sample[label_index][3][0] = sample[1]

                    if (sample[2]== 4 or sample[2]==6 or sample[2]==1):
                        sample[label_index][4][0] = sample[1]+751
                        sample[label_index][4][1] = sample[1]
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 5):
                        sample[label_index][4][1] = sample[1]+751
                        sample[label_index][4][0] = sample[1]

                    if (sample[2]==4 or sample[2]==2 or sample[2]==6 ):
                        sample[label_index][5][0] = sample[1] + 751
                        sample[label_index][5][1] = sample[1]
                    elif(sample[2]==1 or sample[2]==3 or sample[2]==5):
                        sample[label_index][5][1] = sample[1] + 751
                        sample[label_index][5][0] = sample[1]

                    if (sample[2] == 1 or sample[2] == 2 or sample[2] == 6):
                        sample[label_index][6][0] = sample[1] + 751
                        sample[label_index][6][1] = sample[1]
                    elif (sample[2] == 3 or sample[2] == 4 or sample[2] == 5):
                        sample[label_index][6][1] = sample[1] + 751
                        sample[label_index][6][0] = sample[1]

                    if (sample[2]==1 or sample[2]== 2 or sample[2]==3):
                        sample[label_index][7][0] = sample[1] + 751
                        sample[label_index][7][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 5 or sample[2] == 6):
                        sample[label_index][7][1] = sample[1] + 751
                        sample[label_index][7][0] = sample[1]

            elif dataset == 'market' and is_target:
                for sample in samples:
                    if (sample[2]== 1 or sample[2]==4 or sample[2]==5):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif(sample[2]== 2 or sample[2]==3 or sample[2]==6):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if (sample[2]==2 or sample[2]==4 or sample[2]==5 ):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif(sample[2]==3 or sample[2]==1 or sample[2]==6 ):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if (sample[2]==1 or sample[2]==5 or sample[2]==6):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif(sample[2]==4 or sample[2]==2 or sample[2]==3):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if (sample[2]==2 or sample[2]== 5 or sample[2]==1):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif(sample[2]==4 or sample[2]==6 or sample[2]==3):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1

                    if (sample[2]== 4 or sample[2]==6 or sample[2]==1):
                        sample[label_index][4][0] = 1
                        sample[label_index][4][1] = 0
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 5):
                        sample[label_index][4][0] = 0
                        sample[label_index][4][1] = 1

                    if (sample[2]==4 or sample[2]==2 or sample[2]==6 ):
                        sample[label_index][5][0] = 1
                        sample[label_index][5][1] = 0
                    elif(sample[2]==1 or sample[2]==3 or sample[2]==5):
                        sample[label_index][5][0] = 0
                        sample[label_index][5][1] = 1

                    if (sample[2] == 1 or sample[2] == 2 or sample[2] == 6):
                        sample[label_index][6][0] = 1
                        sample[label_index][6][1] = 0
                    elif (sample[2] == 3 or sample[2] == 4 or sample[2] == 5):
                        sample[label_index][6][0] = 0
                        sample[label_index][6][1] = 1

                    if (sample[2]==1 or sample[2]== 2 or sample[2]==3):
                        sample[label_index][7][0] = 1
                        sample[label_index][7][1] = 0
                    elif (sample[2] == 4 or sample[2] == 5 or sample[2] == 6):
                        sample[label_index][7][0] = 0
                        sample[label_index][7][1] = 1

            elif dataset=='msmt' and is_target:
                for sample in samples:
                    if(sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif (sample[2]+1 == 4 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if(sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==4):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif (sample[2]+1 == 2 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if(sample[2]+1==3 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif (sample[2]+1 == 2 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==4):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if(sample[2]+1==1 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==6):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif (sample[2]+1 == 5 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==7 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1

                    if(sample[2]+1==5 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==2):
                        sample[label_index][4][0] = 1
                        sample[label_index][4][1] = 0
                    elif (sample[2]+1 == 9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==8 or sample[2]+1==7 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==6):
                        sample[label_index][4][0] = 0
                        sample[label_index][4][1] = 1

                    if(sample[2]+1==9 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==7):
                        sample[label_index][5][0] = 1
                        sample[label_index][5][1] = 0
                    elif (sample[2]+1 == 10 or sample[2]+1==12 or sample[2]+1==11 or sample[2]+1==5 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==8):
                        sample[label_index][5][0] = 0
                        sample[label_index][5][1] = 1

                    if(sample[2]+1==8 or sample[2]+1==10 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==2 or sample[2]+1==4 or sample[2]+1==7):
                        sample[label_index][6][0] = 1
                        sample[label_index][6][1] = 0
                    elif (sample[2]+1 == 12 or sample[2]+1==13 or sample[2]+1==1 or sample[2]+1==3 or sample[2]+1==5 or sample[2]+1==6 or sample[2]+1==9 or sample[2]+1==11):
                        sample[label_index][6][0] = 0
                        sample[label_index][6][1] = 1

                    if(sample[2]+1==14 or sample[2]+1==1 or sample[2]+1==5 or sample[2]+1==9 or sample[2]+1==11 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][7][0] = 1
                        sample[label_index][7][1] = 0
                    elif (sample[2]+1 == 15 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==12 or sample[2]+1==13 or sample[2]+1==10 or sample[2]+1==2):
                        sample[label_index][7][0] = 0
                        sample[label_index][7][1] = 1

            elif dataset == 'msmt' and is_target==False:
                for sample in samples:
                    if(sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1):
                        sample[label_index][0][0] = sample[1]+1041
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2]+1 == 4 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][0][0] = sample[1]
                        sample[label_index][0][1] = sample[1]+1041

                    if(sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==4):
                        sample[label_index][1][0] = sample[1]+1041
                        sample[label_index][1][1] = sample[1]
                    elif (sample[2]+1 == 2 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][1][0] = sample[1]
                        sample[label_index][1][1] = sample[1]+1041

                    if(sample[2]+1==3 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][2][0] = sample[1]+1041
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2]+1== 2 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==4):
                        sample[label_index][2][0] = sample[1]
                        sample[label_index][2][1] = sample[1]+1041

                    if(sample[2]+1==1 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==6):
                        sample[label_index][3][0] = sample[1]+1041
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2]+1 == 5 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==7 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][3][0] = sample[1]
                        sample[label_index][3][1] = sample[1]+1041

                    if(sample[2]+1==5 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==2):
                        sample[label_index][4][0] = sample[1]+1041
                        sample[label_index][4][1] = sample[1]
                    elif (sample[2]+1 == 9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==8 or sample[2]+1==7 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==6):
                        sample[label_index][4][0] = sample[1]
                        sample[label_index][4][1] = sample[1]+1041

                    if(sample[2]+1==9 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==7):
                        sample[label_index][5][0] = sample[1]+1041
                        sample[label_index][5][1] = sample[1]
                    elif (sample[2]+1 == 10 or sample[2]+1==12 or sample[2]+1==11 or sample[2]+1==5 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==8):
                        sample[label_index][5][0] = sample[1]
                        sample[label_index][5][1] = sample[1]+1041

                    if(sample[2]+1==8 or sample[2]+1==10 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==2 or sample[2]+1==4 or sample[2]+1==7):
                        sample[label_index][6][0] = sample[1]+1041
                        sample[label_index][6][1] = sample[1]
                    elif (sample[2]+1 == 12 or sample[2]+1==13 or sample[2]+1==1 or sample[2]+1==3 or sample[2]+1==5 or sample[2]+1==6 or sample[2]+1==9 or sample[2]+1==11):
                        sample[label_index][6][0] = sample[1]
                        sample[label_index][6][1] = sample[1]+1041

                    if(sample[2]+1==14 or sample[2]+1==1 or sample[2]+1==5 or sample[2]+1==9 or sample[2]+1==11 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][7][0] = sample[1]+1041
                        sample[label_index][7][1] = sample[1]
                    elif (sample[2]+1 == 15 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==12 or sample[2]+1==13 or sample[2]+1==10 or sample[2]+1==2):
                        sample[label_index][7][0] = sample[1]
                        sample[label_index][7][1] = sample[1]+1041
        elif s_t=='duke_msmt' or s_t=='msmt_duke':
            if dataset=='duke' and is_target==False:
                for sample in samples:
                    if (sample[2]== 1 or sample[2]==4 or sample[2]==5 or sample[2]==7):
                        sample[label_index][0][0] = sample[1]+702
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 6 or sample[2]==8):
                        sample[label_index][0][1] = sample[1]+702
                        sample[label_index][0][0] = sample[1]

                    if (sample[2]==2 or sample[2]==4 or sample[2]==5 or sample[2]==8):
                        sample[label_index][1][0] = sample[1] + 702
                        sample[label_index][1][1] = sample[1]
                    elif(sample[2]==3 or sample[2]==1 or sample[2]==6 or sample[2]==7):
                        sample[label_index][1][1] = sample[1] + 702
                        sample[label_index][1][0] = sample[1]

                    if (sample[2] == 1 or sample[2] == 5 or sample[2] == 6 or sample[2]==7):
                        sample[label_index][2][0] = sample[1] + 702
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 2 or sample[2] == 3 or sample[2]==8):
                        sample[label_index][2][1] = sample[1] + 702
                        sample[label_index][2][0] = sample[1]

                    if (sample[2]==2 or sample[2]== 5 or sample[2]==1 or sample[2]==8):
                        sample[label_index][3][0] = sample[1] + 702
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 6 or sample[2] == 3 or sample[2]==7):
                        sample[label_index][3][1] = sample[1] + 702
                        sample[label_index][3][0] = sample[1]

                    if (sample[2]== 4 or sample[2]==6 or sample[2]==1 or sample[2]==8):
                        sample[label_index][4][0] = sample[1]+702
                        sample[label_index][4][1] = sample[1]
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 5 or sample[2]==7):
                        sample[label_index][4][1] = sample[1]+702
                        sample[label_index][4][0] = sample[1]

                    if (sample[2]==4 or sample[2]==2 or sample[2]==6 or sample[2]==7):
                        sample[label_index][5][0] = sample[1] + 702
                        sample[label_index][5][1] = sample[1]
                    elif(sample[2]==1 or sample[2]==3 or sample[2]==5 or sample[2]==8):
                        sample[label_index][5][1] = sample[1] + 702
                        sample[label_index][5][0] = sample[1]

                    if (sample[2] == 1 or sample[2] == 2 or sample[2] == 6 or sample[2]==8):
                        sample[label_index][6][0] = sample[1] + 702
                        sample[label_index][6][1] = sample[1]
                    elif (sample[2] == 3 or sample[2] == 4 or sample[2] == 5 or sample[2]==7):
                        sample[label_index][6][1] = sample[1] + 702
                        sample[label_index][6][0] = sample[1]

                    if (sample[2]==1 or sample[2]== 2 or sample[2]==3 or sample[2]==8):
                        sample[label_index][7][0] = sample[1] + 702
                        sample[label_index][7][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 5 or sample[2] == 6 or sample[2]==7):
                        sample[label_index][7][1] = sample[1] + 702
                        sample[label_index][7][0] = sample[1]

            elif dataset == 'duke' and is_target:
                for sample in samples:
                    if (sample[2]== 1 or sample[2]==4 or sample[2]==5 or sample[2]==7):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif(sample[2]== 2 or sample[2]==3 or sample[2]==6 or sample[2]==8):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if (sample[2]==2 or sample[2]==4 or sample[2]==5 or sample[2]==8 ):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif(sample[2]==3 or sample[2]==1 or sample[2]==6 or sample[2]==7):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if (sample[2]==1 or sample[2]==5 or sample[2]==6 or sample[2]==7):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif(sample[2]==4 or sample[2]==2 or sample[2]==3 or sample[2]==8):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if (sample[2]==2 or sample[2]== 5 or sample[2]==1 or sample[2]==8):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif(sample[2]==4 or sample[2]==6 or sample[2]==3 or sample[2]==7):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1

                    if (sample[2]== 4 or sample[2]==6 or sample[2]==1 or sample[2]==8):
                        sample[label_index][4][0] = 1
                        sample[label_index][4][1] = 0
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 5 or sample[2]==7):
                        sample[label_index][4][0] = 0
                        sample[label_index][4][1] = 1

                    if (sample[2]==4 or sample[2]==2 or sample[2]==6 or sample[2]==7):
                        sample[label_index][5][0] = 1
                        sample[label_index][5][1] = 0
                    elif(sample[2]==1 or sample[2]==3 or sample[2]==5 or sample[2]==8):
                        sample[label_index][5][0] = 0
                        sample[label_index][5][1] = 1

                    if (sample[2] == 1 or sample[2] == 2 or sample[2] == 6 or sample[2]==8):
                        sample[label_index][6][0] = 1
                        sample[label_index][6][1] = 0
                    elif (sample[2] == 3 or sample[2] == 4 or sample[2] == 5or sample[2]==7):
                        sample[label_index][6][0] = 0
                        sample[label_index][6][1] = 1

                    if (sample[2]==1 or sample[2]== 2 or sample[2]==3 or sample[2]==8):
                        sample[label_index][7][0] = 1
                        sample[label_index][7][1] = 0
                    elif (sample[2] == 4 or sample[2] == 5 or sample[2] == 6 or sample[2]==7):
                        sample[label_index][7][0] = 0
                        sample[label_index][7][1] = 1

            elif dataset=='msmt' and is_target:
                for sample in samples:
                    if(sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif (sample[2]+1 == 4 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if(sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==4):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif (sample[2]+1 == 2 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if(sample[2]+1==3 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif (sample[2]+1 == 2 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==4):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if(sample[2]+1==1 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==6):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif (sample[2]+1 == 5 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==7 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1

                    if(sample[2]+1==5 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==2):
                        sample[label_index][4][0] = 1
                        sample[label_index][4][1] = 0
                    elif (sample[2]+1 == 9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==8 or sample[2]+1==7 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==6):
                        sample[label_index][4][0] = 0
                        sample[label_index][4][1] = 1

                    if(sample[2]+1==9 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==7):
                        sample[label_index][5][0] = 1
                        sample[label_index][5][1] = 0
                    elif (sample[2]+1 == 10 or sample[2]+1==12 or sample[2]+1==11 or sample[2]+1==5 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==8):
                        sample[label_index][5][0] = 0
                        sample[label_index][5][1] = 1

                    if(sample[2]+1==8 or sample[2]+1==10 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==2 or sample[2]+1==4 or sample[2]+1==7):
                        sample[label_index][6][0] = 1
                        sample[label_index][6][1] = 0
                    elif (sample[2]+1 == 12 or sample[2]+1==13 or sample[2]+1==1 or sample[2]+1==3 or sample[2]+1==5 or sample[2]+1==6 or sample[2]+1==9 or sample[2]+1==11):
                        sample[label_index][6][0] = 0
                        sample[label_index][6][1] = 1

                    if(sample[2]+1==14 or sample[2]+1==1 or sample[2]+1==5 or sample[2]+1==9 or sample[2]+1==11 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][7][0] = 1
                        sample[label_index][7][1] = 0
                    elif (sample[2]+1 == 15 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==12 or sample[2]+1==13 or sample[2]+1==10 or sample[2]+1==2):
                        sample[label_index][7][0] = 0
                        sample[label_index][7][1] = 1

            elif dataset == 'msmt' and is_target==False:
                for sample in samples:
                    if(sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1):
                        sample[label_index][0][0] = sample[1]+1041
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2]+1 == 4 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][0][0] = sample[1]
                        sample[label_index][0][1] = sample[1]+1041

                    if(sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==4):
                        sample[label_index][1][0] = sample[1]+1041
                        sample[label_index][1][1] = sample[1]
                    elif (sample[2]+1 == 2 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][1][0] = sample[1]
                        sample[label_index][1][1] = sample[1]+1041

                    if(sample[2]+1==3 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][2][0] = sample[1]+1041
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2]+1== 2 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==4):
                        sample[label_index][2][0] = sample[1]
                        sample[label_index][2][1] = sample[1]+1041

                    if(sample[2]+1==1 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==6):
                        sample[label_index][3][0] = sample[1]+1041
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2]+1 == 5 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==7 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][3][0] = sample[1]
                        sample[label_index][3][1] = sample[1]+1041

                    if(sample[2]+1==5 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==2):
                        sample[label_index][4][0] = sample[1]+1041
                        sample[label_index][4][1] = sample[1]
                    elif (sample[2]+1 == 9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==8 or sample[2]+1==7 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==6):
                        sample[label_index][4][0] = sample[1]
                        sample[label_index][4][1] = sample[1]+1041

                    if(sample[2]+1==9 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==7):
                        sample[label_index][5][0] = sample[1]+1041
                        sample[label_index][5][1] = sample[1]
                    elif (sample[2]+1 == 10 or sample[2]+1==12 or sample[2]+1==11 or sample[2]+1==5 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==8):
                        sample[label_index][5][0] = sample[1]
                        sample[label_index][5][1] = sample[1]+1041

                    if(sample[2]+1==8 or sample[2]+1==10 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==2 or sample[2]+1==4 or sample[2]+1==7):
                        sample[label_index][6][0] = sample[1]+1041
                        sample[label_index][6][1] = sample[1]
                    elif (sample[2]+1 == 12 or sample[2]+1==13 or sample[2]+1==1 or sample[2]+1==3 or sample[2]+1==5 or sample[2]+1==6 or sample[2]+1==9 or sample[2]+1==11):
                        sample[label_index][6][0] = sample[1]
                        sample[label_index][6][1] = sample[1]+1041

                    if(sample[2]+1==14 or sample[2]+1==1 or sample[2]+1==5 or sample[2]+1==9 or sample[2]+1==11 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][7][0] = sample[1]+1041
                        sample[label_index][7][1] = sample[1]
                    elif (sample[2]+1 == 15 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==12 or sample[2]+1==13 or sample[2]+1==10 or sample[2]+1==2):
                        sample[label_index][7][0] = sample[1]
                        sample[label_index][7][1] = sample[1]+1041
        elif s_t=='msmt_grid':
            if dataset=='grid' and is_target==False:
                for sample in samples:
                    if (sample[2]== 1 or sample[2]==4 or sample[2]==5 or sample[2]==7):
                        sample[label_index][0][0] = sample[1]+702
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 6 or sample[2]==8):
                        sample[label_index][0][1] = sample[1]+702
                        sample[label_index][0][0] = sample[1]

                    if (sample[2]==2 or sample[2]==4 or sample[2]==5 or sample[2]==8):
                        sample[label_index][1][0] = sample[1] + 702
                        sample[label_index][1][1] = sample[1]
                    elif(sample[2]==3 or sample[2]==1 or sample[2]==6 or sample[2]==7):
                        sample[label_index][1][1] = sample[1] + 702
                        sample[label_index][1][0] = sample[1]

                    if (sample[2] == 1 or sample[2] == 5 or sample[2] == 6 or sample[2]==7):
                        sample[label_index][2][0] = sample[1] + 702
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 2 or sample[2] == 3 or sample[2]==8):
                        sample[label_index][2][1] = sample[1] + 702
                        sample[label_index][2][0] = sample[1]

                    if (sample[2]==2 or sample[2]== 5 or sample[2]==1 or sample[2]==8):
                        sample[label_index][3][0] = sample[1] + 702
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 6 or sample[2] == 3 or sample[2]==7):
                        sample[label_index][3][1] = sample[1] + 702
                        sample[label_index][3][0] = sample[1]

                    if (sample[2]== 4 or sample[2]==6 or sample[2]==1 or sample[2]==8):
                        sample[label_index][4][0] = sample[1]+702
                        sample[label_index][4][1] = sample[1]
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 5 or sample[2]==7):
                        sample[label_index][4][1] = sample[1]+702
                        sample[label_index][4][0] = sample[1]

                    if (sample[2]==4 or sample[2]==2 or sample[2]==6 or sample[2]==7):
                        sample[label_index][5][0] = sample[1] + 702
                        sample[label_index][5][1] = sample[1]
                    elif(sample[2]==1 or sample[2]==3 or sample[2]==5 or sample[2]==8):
                        sample[label_index][5][1] = sample[1] + 702
                        sample[label_index][5][0] = sample[1]

                    if (sample[2] == 1 or sample[2] == 2 or sample[2] == 6 or sample[2]==8):
                        sample[label_index][6][0] = sample[1] + 702
                        sample[label_index][6][1] = sample[1]
                    elif (sample[2] == 3 or sample[2] == 4 or sample[2] == 5 or sample[2]==7):
                        sample[label_index][6][1] = sample[1] + 702
                        sample[label_index][6][0] = sample[1]

                    if (sample[2]==1 or sample[2]== 2 or sample[2]==3 or sample[2]==8):
                        sample[label_index][7][0] = sample[1] + 702
                        sample[label_index][7][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 5 or sample[2] == 6 or sample[2]==7):
                        sample[label_index][7][1] = sample[1] + 702
                        sample[label_index][7][0] = sample[1]

            elif dataset == 'grid' and is_target:
                for sample in samples:
                    if (sample[2]+1== 1 or sample[2]+1==4 or sample[2]+1==5 or sample[2]+1==7):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif(sample[2]+1== 2 or sample[2]+1==3 or sample[2]+1==6 or sample[2]+1==8):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if (sample[2]+1==2 or sample[2]+1==4 or sample[2]+1==5 or sample[2]+1==8 ):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif(sample[2]+1==3 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==7):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if (sample[2]+1==1 or sample[2]+1==5 or sample[2]+1==6 or sample[2]+1==7):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif(sample[2]+1==4 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==8):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if (sample[2]+1==2 or sample[2]+1== 5 or sample[2]+1==1 or sample[2]+1==8):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif(sample[2]+1==4 or sample[2]+1==6 or sample[2]+1==3 or sample[2]+1==7):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1

                    if (sample[2]+1== 4 or sample[2]+1==6 or sample[2]+1==1 or sample[2]+1==8):
                        sample[label_index][4][0] = 1
                        sample[label_index][4][1] = 0
                    elif (sample[2]+1 == 2 or sample[2]+1 == 3 or sample[2]+1 == 5 or sample[2]+1==7):
                        sample[label_index][4][0] = 0
                        sample[label_index][4][1] = 1

                    if (sample[2]+1==4 or sample[2]+1==2 or sample[2]+1==6 or sample[2]+1==7):
                        sample[label_index][5][0] = 1
                        sample[label_index][5][1] = 0
                    elif(sample[2]+1==1 or sample[2]+1==3 or sample[2]+1==5 or sample[2]+1==8):
                        sample[label_index][5][0] = 0
                        sample[label_index][5][1] = 1

                    if (sample[2]+1 == 1 or sample[2]+1 == 2 or sample[2]+1== 6 or sample[2]+1==8):
                        sample[label_index][6][0] = 1
                        sample[label_index][6][1] = 0
                    elif (sample[2]+1 == 3 or sample[2]+1 == 4 or sample[2]+1 == 5or sample[2]+1==7):
                        sample[label_index][6][0] = 0
                        sample[label_index][6][1] = 1

                    if (sample[2]+1==1 or sample[2]+1== 2 or sample[2]+1==3 or sample[2]+1==8):
                        sample[label_index][7][0] = 1
                        sample[label_index][7][1] = 0
                    elif (sample[2]+1 == 4 or sample[2]+1 == 5 or sample[2]+1 == 6 or sample[2]+1==7):
                        sample[label_index][7][0] = 0
                        sample[label_index][7][1] = 1

            elif dataset=='msmt' and is_target:
                for sample in samples:
                    if(sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif (sample[2]+1 == 4 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if(sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==4):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif (sample[2]+1 == 2 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if(sample[2]+1==3 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif (sample[2]+1 == 2 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==4):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if(sample[2]+1==1 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==6):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif (sample[2]+1 == 5 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==7 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1

                    if(sample[2]+1==5 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==2):
                        sample[label_index][4][0] = 1
                        sample[label_index][4][1] = 0
                    elif (sample[2]+1 == 9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==8 or sample[2]+1==7 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==6):
                        sample[label_index][4][0] = 0
                        sample[label_index][4][1] = 1

                    if(sample[2]+1==9 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==7):
                        sample[label_index][5][0] = 1
                        sample[label_index][5][1] = 0
                    elif (sample[2]+1 == 10 or sample[2]+1==12 or sample[2]+1==11 or sample[2]+1==5 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==8):
                        sample[label_index][5][0] = 0
                        sample[label_index][5][1] = 1

                    if(sample[2]+1==8 or sample[2]+1==10 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==2 or sample[2]+1==4 or sample[2]+1==7):
                        sample[label_index][6][0] = 1
                        sample[label_index][6][1] = 0
                    elif (sample[2]+1 == 12 or sample[2]+1==13 or sample[2]+1==1 or sample[2]+1==3 or sample[2]+1==5 or sample[2]+1==6 or sample[2]+1==9 or sample[2]+1==11):
                        sample[label_index][6][0] = 0
                        sample[label_index][6][1] = 1

                    if(sample[2]+1==14 or sample[2]+1==1 or sample[2]+1==5 or sample[2]+1==9 or sample[2]+1==11 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][7][0] = 1
                        sample[label_index][7][1] = 0
                    elif (sample[2]+1 == 15 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==12 or sample[2]+1==13 or sample[2]+1==10 or sample[2]+1==2):
                        sample[label_index][7][0] = 0
                        sample[label_index][7][1] = 1

            elif dataset == 'msmt' and is_target==False:
                for sample in samples:
                    if(sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1):
                        sample[label_index][0][0] = sample[1]+1041
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2]+1 == 4 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][0][0] = sample[1]
                        sample[label_index][0][1] = sample[1]+1041

                    if(sample[2]+1==3 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==4):
                        sample[label_index][1][0] = sample[1]+1041
                        sample[label_index][1][1] = sample[1]
                    elif (sample[2]+1 == 2 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][1][0] = sample[1]
                        sample[label_index][1][1] = sample[1]+1041

                    if(sample[2]+1==3 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15):
                        sample[label_index][2][0] = sample[1]+1041
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2]+1== 2 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==5 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==4):
                        sample[label_index][2][0] = sample[1]
                        sample[label_index][2][1] = sample[1]+1041

                    if(sample[2]+1==1 or sample[2]+1==8 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==6):
                        sample[label_index][3][0] = sample[1]+1041
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2]+1 == 5 or sample[2]+1==9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==7 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][3][0] = sample[1]
                        sample[label_index][3][1] = sample[1]+1041

                    if(sample[2]+1==5 or sample[2]+1==11 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==2):
                        sample[label_index][4][0] = sample[1]+1041
                        sample[label_index][4][1] = sample[1]
                    elif (sample[2]+1 == 9 or sample[2]+1==10 or sample[2]+1==12 or sample[2]+1==8 or sample[2]+1==7 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==6):
                        sample[label_index][4][0] = sample[1]
                        sample[label_index][4][1] = sample[1]+1041

                    if(sample[2]+1==9 or sample[2]+1==13 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==1 or sample[2]+1==6 or sample[2]+1==7):
                        sample[label_index][5][0] = sample[1]+1041
                        sample[label_index][5][1] = sample[1]
                    elif (sample[2]+1 == 10 or sample[2]+1==12 or sample[2]+1==11 or sample[2]+1==5 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==4 or sample[2]+1==8):
                        sample[label_index][5][0] = sample[1]
                        sample[label_index][5][1] = sample[1]+1041

                    if(sample[2]+1==8 or sample[2]+1==10 or sample[2]+1==14 or sample[2]+1==15 or sample[2]+1==2 or sample[2]+1==4 or sample[2]+1==7):
                        sample[label_index][6][0] = sample[1]+1041
                        sample[label_index][6][1] = sample[1]
                    elif (sample[2]+1 == 12 or sample[2]+1==13 or sample[2]+1==1 or sample[2]+1==3 or sample[2]+1==5 or sample[2]+1==6 or sample[2]+1==9 or sample[2]+1==11):
                        sample[label_index][6][0] = sample[1]
                        sample[label_index][6][1] = sample[1]+1041

                    if(sample[2]+1==14 or sample[2]+1==1 or sample[2]+1==5 or sample[2]+1==9 or sample[2]+1==11 or sample[2]+1==3 or sample[2]+1==4):
                        sample[label_index][7][0] = sample[1]+1041
                        sample[label_index][7][1] = sample[1]
                    elif (sample[2]+1 == 15 or sample[2]+1==6 or sample[2]+1==7 or sample[2]+1==8 or sample[2]+1==12 or sample[2]+1==13 or sample[2]+1==10 or sample[2]+1==2):
                        sample[label_index][7][0] = sample[1]
                        sample[label_index][7][1] = sample[1]+1041
        elif s_t=='market_grid':
            if dataset=='market' and is_target==False:
                for sample in samples:
                    if (sample[2]== 1 or sample[2]==4 or sample[2]==5):
                        sample[label_index][0][0] = sample[1]+751
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 6):
                        sample[label_index][0][1] = sample[1]+751
                        sample[label_index][0][0] = sample[1]

                    if (sample[2]==2 or sample[2]==4 or sample[2]==5 ):
                        sample[label_index][1][0] = sample[1] + 751
                        sample[label_index][1][1] = sample[1]
                    elif(sample[2]==3 or sample[2]==1 or sample[2]==6):
                        sample[label_index][1][1] = sample[1] + 751
                        sample[label_index][1][0] = sample[1]

                    if (sample[2] == 1 or sample[2] == 5 or sample[2] == 6):
                        sample[label_index][2][0] = sample[1] + 751
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 2 or sample[2] == 3):
                        sample[label_index][2][1] = sample[1] + 751
                        sample[label_index][2][0] = sample[1]

                    if (sample[2]==2 or sample[2]== 5 or sample[2]==1):
                        sample[label_index][3][0] = sample[1] + 751
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 6 or sample[2] == 3):
                        sample[label_index][3][1] = sample[1] + 751
                        sample[label_index][3][0] = sample[1]
                return samples


            elif dataset=='grid' and is_target:
                for sample in samples:
                    if(sample[2]+1==5 or sample[2]+1==7 or sample[2]+1==6 or sample[2]+1==4):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif (sample[2]+1 == 1 or sample[2]+1 == 2 or sample[2]+1 == 3 or sample[2]+1 == 8):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if(sample[2]+1==5 or sample[2]+1==2 or sample[2]+1==3 or sample[2]+1==8):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif (sample[2]+1 == 7 or sample[2]+1 == 6 or sample[2]+1 == 4 or sample[2]+1 == 1):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if (sample[2]+1==7 or sample[2]+1==3 or sample[2]+1==8 or sample[2]+1==1):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif (sample[2]+1 == 5 or sample[2]+1 == 4 or sample[2]+1 == 6 or sample[2]+1 == 2):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if (sample[2]+1==6 or sample[2]+1==3 or sample[2]+1==5 or sample[2]+1==1):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif (sample[2]+1 == 4 or sample[2]+1 == 8 or sample[2]+1 == 2 or sample[2]+1 == 7):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1
        elif s_t == 'duke_grid':
            if dataset == 'grid' and is_target:
                for sample in samples:
                    if (sample[2]+1 == 1 or sample[2]+1 == 4 or sample[2]+1 == 5):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif (sample[2]+1 == 2 or sample[2]+1 == 3 or sample[2]+1 == 6):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if (sample[2]+1 == 2 or sample[2]+1 == 4 or sample[2]+1 == 5):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif (sample[2]+1 == 3 or sample[2]+1 == 1 or sample[2]+1 == 6):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if (sample[2]+1 == 1 or sample[2]+1 == 5 or sample[2]+1 == 6):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif (sample[2]+1 == 4 or sample[2]+1 == 2 or sample[2]+1 == 3):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if (sample[2]+1 == 2 or sample[2]+1 == 5 or sample[2]+1 == 1):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif (sample[2]+1 == 4 or sample[2]+1 == 6 or sample[2]+1 == 3):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1

            elif dataset == 'duke' and is_target == False:
                for sample in samples:
                    if (sample[2] == 5 or sample[2] == 7 or sample[2] == 6 or sample[2] == 4):
                        sample[label_index][0][0] = sample[1] + 702
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2] == 1 or sample[2] == 2 or sample[2] == 3 or sample[2] == 8):
                        sample[label_index][0][1] = sample[1] + 702
                        sample[label_index][0][0] = sample[1]

                    if (sample[2] == 5 or sample[2] == 2 or sample[2] == 3 or sample[2] == 8):
                        sample[label_index][1][0] = sample[1] + 702
                        sample[label_index][1][1] = sample[1]
                    elif (sample[2] == 7 or sample[2] == 6 or sample[2] == 4 or sample[2] == 1):
                        sample[label_index][1][1] = sample[1] + 702
                        sample[label_index][1][0] = sample[1]

                    if (sample[2] == 7 or sample[2] == 3 or sample[2] == 8 or sample[2] == 1):
                        sample[label_index][2][0] = sample[1] + 702
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2] == 5 or sample[2] == 4 or sample[2] == 6 or sample[2] == 2):
                        sample[label_index][2][1] = sample[1] + 702
                        sample[label_index][2][0] = sample[1]

                    if (sample[2] == 6 or sample[2] == 3 or sample[2] == 5 or sample[2] == 1):
                        sample[label_index][3][0] = sample[1] + 702
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 8 or sample[2] == 2 or sample[2] == 7):
                        sample[label_index][3][1] = sample[1] + 702
                        sample[label_index][3][0] = sample[1]
        elif s_t=='market_prid':
            if dataset=='market' and is_target==False:
                for sample in samples:
                    if (sample[2]== 1 or sample[2]==4 or sample[2]==5):
                        sample[label_index][0][0] = sample[1]+751
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2] == 2 or sample[2] == 3 or sample[2] == 6):
                        sample[label_index][0][1] = sample[1]+751
                        sample[label_index][0][0] = sample[1]

                    if (sample[2]==2 or sample[2]==4 or sample[2]==5 ):
                        sample[label_index][1][0] = sample[1] + 751
                        sample[label_index][1][1] = sample[1]
                    elif(sample[2]==3 or sample[2]==1 or sample[2]==6):
                        sample[label_index][1][1] = sample[1] + 751
                        sample[label_index][1][0] = sample[1]

                    if (sample[2] == 1 or sample[2] == 5 or sample[2] == 6):
                        sample[label_index][2][0] = sample[1] + 751
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 2 or sample[2] == 3):
                        sample[label_index][2][1] = sample[1] + 751
                        sample[label_index][2][0] = sample[1]
                return samples
            elif dataset=='prid' and is_target:
                for sample in samples:
                    if(sample[2]+1==2):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif (sample[2]+1 == 1):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if(sample[2]+1==1):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif (sample[2]+1 == 2):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if (sample[2]+1==1):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif (sample[2]+1 == 2):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1
        elif s_t == 'duke_prid':
            if dataset == 'prid' and is_target:
                for sample in samples:
                    if (sample[2]+1 == 1 ):
                        sample[label_index][0][0] = 1
                        sample[label_index][0][1] = 0
                    elif (sample[2]+1 == 2 ):
                        sample[label_index][0][0] = 0
                        sample[label_index][0][1] = 1

                    if (sample[2]+1 == 1):
                        sample[label_index][1][0] = 1
                        sample[label_index][1][1] = 0
                    elif (sample[2]+1 == 2):
                        sample[label_index][1][0] = 0
                        sample[label_index][1][1] = 1

                    if (sample[2]+1 == 1):
                        sample[label_index][2][0] = 1
                        sample[label_index][2][1] = 0
                    elif (sample[2]+1 == 2):
                        sample[label_index][2][0] = 0
                        sample[label_index][2][1] = 1

                    if (sample[2]+1 == 1):
                        sample[label_index][3][0] = 1
                        sample[label_index][3][1] = 0
                    elif (sample[2]+1 == 2):
                        sample[label_index][3][0] = 0
                        sample[label_index][3][1] = 1

            elif dataset == 'duke' and is_target == False:
                for sample in samples:
                    if (sample[2] == 5 or sample[2] == 7 or sample[2] == 6 or sample[2] == 4):
                        sample[label_index][0][0] = sample[1] + 702
                        sample[label_index][0][1] = sample[1]
                    elif (sample[2] == 1 or sample[2] == 2 or sample[2] == 3 or sample[2] == 8):
                        sample[label_index][0][1] = sample[1] + 702
                        sample[label_index][0][0] = sample[1]

                    if (sample[2] == 5 or sample[2] == 2 or sample[2] == 3 or sample[2] == 8):
                        sample[label_index][1][0] = sample[1] + 702
                        sample[label_index][1][1] = sample[1]
                    elif (sample[2] == 7 or sample[2] == 6 or sample[2] == 4 or sample[2] == 1):
                        sample[label_index][1][1] = sample[1] + 702
                        sample[label_index][1][0] = sample[1]

                    if (sample[2] == 7 or sample[2] == 3 or sample[2] == 8 or sample[2] == 1):
                        sample[label_index][2][0] = sample[1] + 702
                        sample[label_index][2][1] = sample[1]
                    elif (sample[2] == 5 or sample[2] == 4 or sample[2] == 6 or sample[2] == 2):
                        sample[label_index][2][1] = sample[1] + 702
                        sample[label_index][2][0] = sample[1]

                    if (sample[2] == 6 or sample[2] == 3 or sample[2] == 5 or sample[2] == 1):
                        sample[label_index][3][0] = sample[1] + 702
                        sample[label_index][3][1] = sample[1]
                    elif (sample[2] == 4 or sample[2] == 8 or sample[2] == 2 or sample[2] == 7):
                        sample[label_index][3][1] = sample[1] + 702
                        sample[label_index][3][0] = sample[1]




        return samples


    def _load_images_path(self, folder_dir,dataset='market',make_c_net_id=False,is_target=False):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        if is_target:
            domain_label = 1
        else:
            domain_label = 0
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_file_name(file_name)
                if make_c_net_id:
                    # c_net_id = self._make_camera_net_pid(dataset,identi_id,camera_id,is_target)
                    c_s = [[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id]]
                    t_s = [[camera_id,camera_id],[camera_id,camera_id],[camera_id,camera_id],[camera_id,camera_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id]]
                    samples.append([root_path + file_name, identi_id, camera_id,domain_label,c_s,t_s])
                else:
                    samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _make_camera_net_pid(self,dataset,pid,c_id,is_target=False):
        '''
        :param pid: person id
        :param c_id: camera id
        :return:
        '''
        if dataset=='market':
            pid_num = 751
        elif dataset=='duke':
            if is_target:
                pid_num = 16522
            else:
                pid_num = 702
        if dataset == 'market':
            if(c_id>=3):
                c_net_id = pid +pid_num
                return c_net_id
            else:
                return pid
        elif dataset=='duke':
            if (c_id >= 4):
                c_net_id = pid + pid_num
                return c_net_id
            else:
                return pid

    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0844_c3s2_107328_01.jpg
        :return: 0844, 3
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

    def _show_info(self, train, query, gallery, name=None):
        def analyze(samples):
            pid_num = len(set([sample[1] for sample in samples]))
            cid_num = len(set([sample[2] for sample in samples]))
            sample_num = len(samples)
            return sample_num, pid_num, cid_num

        train_info = analyze(train)
        query_info = analyze(query)
        gallery_info = analyze(gallery)

        # please kindly install prettytable: ```pip install prettyrable```
        table = PrettyTable(['set', 'images', 'identities', 'cameras'])
        table.add_row([self.__class__.__name__ if name is None else name, '', '', ''])
        table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2])])
        table.add_row(['query', str(query_info[0]), str(query_info[1]), str(query_info[2])])
        table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2])])
        print(table)



class GRID(PersonReIDSamples):
    """GRID.

    Reference:
        Loy et al. Multi-camera activity correlation analysis. CVPR 2009.

    URL: `<http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html>`_

    Dataset statistics:
        - identities: 250.
        - images: 1275.
        - cameras: 8.
    """
    dataset_dir = 'GRID/underground_reid'
    dataset_url = 'http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip'
    def __init__(self,root='',is_target=False,train_st='',relabel=True, split_id=6, **kwargs):
        super(GRID, self).__init__()
        # self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.images_dir = osp.join(root, 'images')
        # self.download_dataset(self.dataset_dir, self.dataset_url)

        self.probe_path = osp.join(
            self.dataset_dir,  'probe'
        )
        self.gallery_path = osp.join(
            self.dataset_dir,  'gallery'
        )
        self.split_mat_path = osp.join(
            self.dataset_dir,  'features_and_partitions.mat'
        )
        self.split_path = osp.join(self.dataset_dir, 'splits.json')

        required_files = [
            self.dataset_dir, self.probe_path, self.gallery_path,
            self.split_mat_path
        ]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, '
                'but expected between 0 and {}'.format(
                    split_id,
                    len(splits) - 1
                )
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']
        train = self._relabels_c(train, 4,'grid',train_st,is_target)
        self.train = [tuple(item) for item in train]
        self.query = [tuple(item) for item in query]
        self.gallery = [tuple(item) for item in gallery]
        self.trainval = train
        # self.train = [tuple(item) for item in train]
        # self.query = [tuple(item) for item in query]
        # self.gallery = [tuple(item) for item in gallery]

        print("=> GRID loaded")
        self.print_dataset_statistics(self.train, self.query, self.gallery)
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_trainval_ids, self.num_trainval_imgs, self.num_trainval_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)




    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating 10 random splits')
            split_mat = loadmat(self.split_mat_path)
            trainIdxAll = split_mat['trainIdxAll'][0]  # length = 10
            probe_img_paths = sorted(
                glob.glob(osp.join(self.probe_path, '*.jpeg'))
            )
            gallery_img_paths = sorted(
                glob.glob(osp.join(self.gallery_path, '*.jpeg'))
            )

            splits = []
            for split_idx in range(10):
                train_idxs = trainIdxAll[split_idx][0][0][2][0].tolist()
                assert len(train_idxs) == 125
                idx2label = {
                    idx: label
                    for label, idx in enumerate(train_idxs)
                }

                train, query, gallery = [], [], []

                # processing probe folder
                query_id_list = []
                for img_path in probe_img_paths:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1  # index starts from 0
                    if img_idx in train_idxs:
                        identi_id=idx2label[img_idx]
                        camera_id =camid
                        c_s = [[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id]]
                        t_s = [[camera_id,camera_id],[camera_id,camera_id],[camera_id,camera_id],[camera_id,camera_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id]]
                        # samples.append([root_path + file_name, identi_id, camera_id,domain_label,c_s,t_s])
                        train.append((img_path,identi_id, camera_id))
                    else:
                        query_id_list.append(img_idx)
                        query.append((img_path, img_idx, camid))

                # process gallery folder
                interfere = 0
                for img_path in gallery_img_paths:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1  # index starts from 0
                    if img_idx in train_idxs:
                        identi_id=idx2label[img_idx]
                        camera_id =camid
                        c_s = [[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id]]
                        t_s = [[camera_id,camera_id],[camera_id,camera_id],[camera_id,camera_id],[camera_id,camera_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id]]
                        # samples.append([root_path + file_name, identi_id, camera_id,domain_label,c_s,t_s])
                        train.append((img_path,identi_id, camera_id))
                    else:
                        if img_idx in query_id_list:
                            gallery.append((img_path, img_idx, camid))
                        else:
                            interfere+=1
                            if interfere<401:
                                identi_id=0
                                camera_id =camid
                                c_s = [[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id]]
                                t_s = [[camera_id,camera_id],[camera_id,camera_id],[camera_id,camera_id],[camera_id,camera_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id]]
                                # samples.append([root_path + file_name, identi_id, camera_id,domain_label,c_s,t_s])
                                train.append((img_path,identi_id, camera_id))
                            else:
                                gallery.append((img_path, img_idx, camid))

                split = {
                    'train': train,
                    'query': query,
                    'gallery': gallery,
                    'num_train_pids': 125,
                    'num_query_pids': 125,
                    'num_gallery_pids': 900
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))


    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams


# from __future__ import division, print_function, absolute_import
# import glob
# import os.path as osp
# from scipy.io import loadmat
#
# from ..utils.data import Dataset
# import errno
# import json
# import os
#
#
#
# def mkdir_if_missing(directory):
#     if not osp.exists(directory):
#         try:
#             os.makedirs(directory)
#         except OSError as e:
#             if e.errno != errno.EEXIST:
#                 raise
#
#
# def check_isfile(path):
#     isfile = osp.isfile(path)
#     if not isfile:
#         print("=> Warning: no file found at '{}' (ignored)".format(path))
#     return isfile
#
# def read_json(fpath):
#     with open(fpath, 'r') as f:
#         obj = json.load(f)
#     return obj
#
#
# def write_json(obj, fpath):
#     mkdir_if_missing(osp.dirname(fpath))
#     with open(fpath, 'w') as f:
#         json.dump(obj, f, indent=4, separators=(',', ': '))
#
# class GRID(Dataset):
#     """GRID.
#
#     Reference:
#         Loy et al. Multi-camera activity correlation analysis. CVPR 2009.
#
#     URL: `<http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html>`_
#
#     Dataset statistics:
#         - identities: 250.
#         - images: 1275.
#         - cameras: 8.
#     """
#     dataset_dir = 'GRID'
#     dataset_url = 'http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip'
#
#     def __init__(self, root='', split_id=0, **kwargs):
#         super(GRID, self).__init__(root)
#         # self.root = osp.abspath(osp.expanduser(root))
#         self.dataset_dir = root
#         self.images_dir = osp.join(root, 'images')
#         # self.download_dataset(self.dataset_dir, self.dataset_url)
#
#         self.probe_path = osp.join(
#             self.dataset_dir,  'probe'
#         )
#         self.gallery_path = osp.join(
#             self.dataset_dir,  'gallery'
#         )
#         self.split_mat_path = osp.join(
#             self.dataset_dir,  'features_and_partitions.mat'
#         )
#         self.split_path = osp.join(self.dataset_dir, 'splits.json')
#         # self.images_dir = None
#
#         required_files = [
#             self.dataset_dir, self.probe_path, self.gallery_path,
#             self.split_mat_path
#         ]
#
#         self.check_before_run(required_files)
#
#         self.prepare_split()
#         splits = read_json(self.split_path)
#         if split_id >= len(splits):
#             raise ValueError(
#                 'split_id exceeds range, received {}, '
#                 'but expected between 0 and {}'.format(
#                     split_id,
#                     len(splits) - 1
#                 )
#             )
#         split = splits[split_id]
#
#         train = split['train']
#         query = split['query']
#         gallery = split['gallery']
#
#         self.train = [tuple(item) for item in train]
#         self.query = [tuple(item) for item in query]
#         self.gallery = [tuple(item) for item in gallery]
#
#         print("=> GRID loaded")
#
#         self.print_dataset_statistics(self.train, self.query, self.gallery)
#
#         self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
#         self.trainval = self.train
#
#
#     def print_dataset_statistics(self, train, query, gallery):
#         num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
#         num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
#         num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)
#
#         print("Dataset statistics:")
#         print("  ----------------------------------------")
#         print("  subset   | # ids | # images | # cameras")
#         print("  ----------------------------------------")
#         print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
#         print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
#         print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
#         print("  ----------------------------------------")
#
#     def get_imagedata_info(self, data):
#         pids, cams = [], []
#         for _, pid, camid in data:
#             pids += [pid]
#             cams += [camid]
#         pids = set(pids)
#         cams = set(cams)
#         num_pids = len(pids)
#         num_cams = len(cams)
#         num_imgs = len(data)
#         return num_pids, num_imgs, num_cams
#
#     def check_before_run(self, required_files):
#         """Checks if required files exist before going deeper.
#
#         Args:
#             required_files (str or list): string file name(s).
#         """
#         if isinstance(required_files, str):
#             required_files = [required_files]
#
#         for fpath in required_files:
#             if not osp.exists(fpath):
#                 raise RuntimeError('"{}" is not found'.format(fpath))
#
#     def prepare_split(self):
#         if not osp.exists(self.split_path):
#             print('Creating 10 random splits')
#             split_mat = loadmat(self.split_mat_path)
#             trainIdxAll = split_mat['trainIdxAll'][0]  # length = 10
#             probe_img_paths = sorted(
#                 glob.glob(osp.join(self.probe_path, '*.jpeg'))
#             )
#             gallery_img_paths = sorted(
#                 glob.glob(osp.join(self.gallery_path, '*.jpeg'))
#             )
#
#             splits = []
#             for split_idx in range(10):
#                 train_idxs = trainIdxAll[split_idx][0][0][2][0].tolist()
#                 assert len(train_idxs) == 125
#                 idx2label = {
#                     idx: label
#                     for label, idx in enumerate(train_idxs)
#                 }
#
#                 train, query, gallery = [], [], []
#
#                 # processing probe folder
#                 for img_path in probe_img_paths:
#                     img_name = osp.basename(img_path)
#                     img_idx = int(img_name.split('_')[0])
#                     camid = int(
#                         img_name.split('_')[1]
#                     ) - 1  # index starts from 0
#                     if img_idx in train_idxs:
#                         train.append((img_path, idx2label[img_idx], camid))
#                     else:
#                         query.append((img_path, img_idx, camid))
#
#                 # process gallery folder
#                 for img_path in gallery_img_paths:
#                     img_name = osp.basename(img_path)
#                     img_idx = int(img_name.split('_')[0])
#                     camid = int(
#                         img_name.split('_')[1]
#                     ) - 1  # index starts from 0
#                     if img_idx in train_idxs:
#                         train.append((img_path, idx2label[img_idx], camid))
#                     else:
#                         gallery.append((img_path, img_idx, camid))
#
#                 split = {
#                     'train': train,
#                     'query': query,
#                     'gallery': gallery,
#                     'num_train_pids': 125,
#                     'num_query_pids': 125,
#                     'num_gallery_pids': 900
#                 }
#                 splits.append(split)
#
#             print('Totally {} splits are created'.format(len(splits)))
#             write_json(splits, self.split_path)
#             print('Split file saved to {}'.format(self.split_path))
