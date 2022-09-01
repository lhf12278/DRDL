# -*- coding: utf-8 -*-
# @Author   : Kaixiong Xu
# @Time     : 2020/10/19 17:03
# @contact: xukaixiong@stu.kust.edu.cn
"""
Our partition protocol:
    100 image pairs of the pedestrians appearing under cameras A and B, and 300 interference images are selected as the training set.
    The remaining images of 100 pedestrians captured by camera A are used as the query set, and the remaining images of 100 pedestrians
    captured by camera B together with the rest 249 interference images are used as the gallery set.

    training set: 500 images with 400 identities
    query set: 100 images with 100 identities
    gallery set : 349 images with 349 identities
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
        :param pid: 身份标签
        :param c_id: 相机标签
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


class PRID(PersonReIDSamples):
    """PRID (single-shot version of prid-2011)

    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and Discriminative
        Classification. SCIA 2011.

    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_

    Dataset statistics:
        - Two views.
        - View A captures 385 identities.
        - View B captures 749 identities.
        - 200 identities appear in both views.
    """
    dataset_dir = 'prid2011'
    dataset_url = None

    def __init__(self, root='',is_target=False,train_st='',relabel=True, split_id=3, **kwargs):
        super(PRID, self).__init__()
        # self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.images_dir = osp.join(root, 'images')
        # self.download_dataset(self.dataset_dir, self.dataset_url)

        self.cam_a_dir = osp.join(
            self.dataset_dir,  'single_shot', 'cam_a'
        )
        self.cam_b_dir = osp.join(
            self.dataset_dir,  'single_shot', 'cam_b'
        )
        self.split_path = osp.join(self.dataset_dir, 'splits_single_shot.json')

        required_files = [self.dataset_dir, self.cam_a_dir, self.cam_b_dir]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                    .format(split_id,
                            len(splits) - 1)
            )
        split = splits[split_id]
        # self.show_summary()
        train, query, gallery = self.process_split(split)
        train = self._relabels_c(train, 4,'prid',train_st,is_target)
        self.train = train
        self.query = query
        self.gallery = gallery
        self.trainval = train

        print("=> PRID loaded")
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
            print('Creating splits ...')

            splits = []
            for _ in range(10):
                # randomly sample 100 IDs for train and use the rest 100 IDs for test
                # (note: there are only 200 IDs appearing in both views)
                pids = [i for i in range(1, 201)]
                train_pids = random.sample(pids, 100)
                train_pids.sort()
                test_pids = [i for i in pids if i not in train_pids]
                split = {'train': train_pids, 'test': test_pids}
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file is saved to {}'.format(self.split_path))

    def process_split(self, split):
        train_pids = split['train']
        test_pids = split['test']

        train_pid2label = {pid: label for label, pid in enumerate(train_pids)}

        # train
        train = []
        for pid in train_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            pid = train_pid2label[pid]
            img_a_path = osp.join(self.cam_a_dir, img_name)
            # c_s = [[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid]]
            # t_s = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
            # train.append([img_a_path, pid, 0,0,c_s,t_s])
            train.append((img_a_path, pid, 0))


            img_b_path = osp.join(self.cam_b_dir, img_name)
            # c_s = [[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid]]
            # t_s = [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
            # train.append([img_b_path, pid, 1,1,c_s,t_s])
            train.append((img_b_path, pid, 1))

            # identi_id=train_pid2label[pid]
            # camera_id =camid
            # c_s = [[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id]]
            # t_s = [[camera_id,camera_id],[camera_id,camera_id],[camera_id,camera_id],[camera_id,camera_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id],[identi_id,identi_id]]
            # # samples.append([root_path + file_name, identi_id, camera_id,domain_label,c_s,t_s])
            # train.append((img_path,identi_id, camera_id,camera_id,c_s,t_s))

        # query and gallery
        query, gallery = [], []
        for pid in test_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_a_path = osp.join(self.cam_a_dir, img_name)
            query.append((img_a_path, pid, 0))
            img_b_path = osp.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))
        for pid in range(501, 750):
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_b_path = osp.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))
        for pid in range(201, 501):
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_b_path = osp.join(self.cam_b_dir, img_name)
            c_s = [[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid],[pid,pid]]
            t_s = [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
            train.append((img_b_path, pid, 1))

        return train, query, gallery

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
# import random
# import os.path as osp
#
# from utils.iotools import write_json, read_json
# # from .bases import BaseImageDataset
# from ..utils.data import Dataset
# # from .dataset_loader import ImageDataset_new
#
#
# class PRID(Dataset):
#     """PRID (single-shot version of prid-2011)
#
#     Reference:
#         Hirzer et al. Person Re-Identification by Descriptive and Discriminative
#         Classification. SCIA 2011.
#
#     URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_
#
#     Dataset statistics:
#         - Two views.
#         - View A captures 385 identities.
#         - View B captures 749 identities.
#         - 200 identities appear in both views.
#     """
#     dataset_dir = ''
#     dataset_url = None
#
#     def __init__(self, root='', split_id=1, **kwargs):
#         super(PRID, self).__init__(root)
#         # self.root = osp.abspath(osp.expanduser(root))
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         # self.download_dataset(self.dataset_dir, self.dataset_url)
#
#         self.cam_a_dir = osp.join(
#             self.dataset_dir,  'single_shot', 'cam_a'
#         )
#         self.cam_b_dir = osp.join(
#             self.dataset_dir,  'single_shot', 'cam_b'
#         )
#         self.split_path = osp.join(self.dataset_dir, 'splits_single_shot.json')
#
#         required_files = [self.dataset_dir, self.cam_a_dir, self.cam_b_dir]
#         self.check_before_run(required_files)
#
#         self.prepare_split()
#         splits = read_json(self.split_path)
#         if split_id >= len(splits):
#             raise ValueError(
#                 'split_id exceeds range, received {}, but expected between 0 and {}'
#                     .format(split_id,
#                             len(splits) - 1)
#             )
#         split = splits[split_id]
#         # self.show_summary()
#         train, query, gallery = self.process_split(split)
#         self.train = train
#         self.trainval = train
#         self.query = query
#         self.gallery = gallery
#
#         print("=> PRID loaded")
#         self.print_dataset_statistics(self.train, self.query, self.gallery)
#         self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
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
#             print('Creating splits ...')
#
#             splits = []
#             for _ in range(10):
#                 # randomly sample 100 IDs for train and use the rest 100 IDs for test
#                 # (note: there are only 200 IDs appearing in both views)
#                 pids = [i for i in range(1, 201)]
#                 train_pids = random.sample(pids, 100)
#                 train_pids.sort()
#                 test_pids = [i for i in pids if i not in train_pids]
#                 split = {'train': train_pids, 'test': test_pids}
#                 splits.append(split)
#
#             print('Totally {} splits are created'.format(len(splits)))
#             write_json(splits, self.split_path)
#             print('Split file is saved to {}'.format(self.split_path))
#
#     def process_split(self, split):
#         train_pids = split['train']
#         test_pids = split['test']
#
#         train_pid2label = {pid: label for label, pid in enumerate(train_pids)}
#
#         # train
#         train = []
#         for pid in train_pids:
#             img_name = 'person_' + str(pid).zfill(4) + '.png'
#             pid = train_pid2label[pid]
#             img_a_path = osp.join(self.cam_a_dir, img_name)
#             train.append((img_a_path, pid, 0))
#             img_b_path = osp.join(self.cam_b_dir, img_name)
#             train.append((img_b_path, pid, 1))
#
#         # query and gallery
#         query, gallery = [], []
#         for pid in test_pids:
#             img_name = 'person_' + str(pid).zfill(4) + '.png'
#             img_a_path = osp.join(self.cam_a_dir, img_name)
#             query.append((img_a_path, pid, 0))
#             img_b_path = osp.join(self.cam_b_dir, img_name)
#             gallery.append((img_b_path, pid, 1))
#         for pid in range(201, 750):
#             img_name = 'person_' + str(pid).zfill(4) + '.png'
#             img_b_path = osp.join(self.cam_b_dir, img_name)
#             gallery.append((img_b_path, pid, 1))
#
#         return train, query, gallery
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
