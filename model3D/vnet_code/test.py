# -*- coding: utf-8 -*-
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import vnet
from metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
import losses
from utils import str2bool, count_params
from sklearn.externals import joblib
import SimpleITK as sitk
import imageio

wt_dices = []
tc_dices = []
et_dices = []
wt_sensitivities = []
tc_sensitivities = []
et_sensitivities = []
wt_ppvs = []
tc_ppvs = []
et_ppvs = []
wt_Hausdorf = []
tc_Hausdorf = []
et_Hausdorf = []


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--mode', default=None,
                        help='')

    args = parser.parse_args()

    return args

#获取某个分块的位置信息（0 32 64 96 128）以及 该块属于哪个病例
def GetPatchPosition(PatchPath):
    npName = os.path.basename(PatchPath)
    firstName = npName
    overNum = npName.find(".npy")
    npName = npName[0:overNum]
    PeopleName = npName
    overNum = npName.find("_")
    while(overNum != -1):
        npName = npName[overNum+1:len(npName)]
        overNum = npName.find("_")
    overNum = firstName.find("_"+npName+".npy")
    PeopleName = PeopleName[0:overNum]
    return int(npName),PeopleName

def hausdorff_distance(lT,lP):
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    return hausdorffcomputer.GetAverageHausdorffDistance()#hausdorffcomputer.GetHausdorffDistance()

def CalculateWTTCET(wtpbregion,wtmaskregion,tcpbregion,tcmaskregion,etpbregion,etmaskregion):
    #开始计算WT
    dice = dice_coef(wtpbregion,wtmaskregion)
    wt_dices.append(dice)
    ppv_n = ppv(wtpbregion, wtmaskregion)
    wt_ppvs.append(ppv_n)
    Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
    wt_Hausdorf.append(Hausdorff)
    sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
    wt_sensitivities.append(sensitivity_n)
    # 开始计算TC
    dice = dice_coef(tcpbregion, tcmaskregion)
    tc_dices.append(dice)
    ppv_n = ppv(tcpbregion, tcmaskregion)
    tc_ppvs.append(ppv_n)
    Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
    tc_Hausdorf.append(Hausdorff)
    sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
    tc_sensitivities.append(sensitivity_n)
    # 开始计算ET
    dice = dice_coef(etpbregion, etmaskregion)
    et_dices.append(dice)
    ppv_n = ppv(etpbregion, etmaskregion)
    et_ppvs.append(ppv_n)
    Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
    et_Hausdorf.append(Hausdorff)
    sensitivity_n = sensitivity(etpbregion, etmaskregion)
    et_sensitivities.append(sensitivity_n)

def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %val_args.name)
    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = vnet.__dict__[args.arch](args)

    model = model.cuda()

    # Data loading code
    img_paths = glob(r'D:\Project\CollegeDesign\dataset\BraTs3D\testImage\*')
    mask_paths = glob(r'D:\Project\CollegeDesign\dataset\BraTs3D\testMask\*')

    val_img_paths = img_paths
    val_mask_paths = mask_paths

    #train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    #   train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
    model.eval()

    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    savedir = 'output/%s/'%args.name
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            startFlag = 1
            for mynum, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                input = input.cuda()
                output = model(input)
                output = torch.sigmoid(output).data.cpu().numpy()
                target = target.data.cpu().numpy()
                img_paths = val_img_paths[args.batch_size * mynum:args.batch_size * (mynum + 1)]
                #print(len(val_loader))
                for i in range(output.shape[0]):
                    if (startFlag == 1):#第一个块的处理
                        startFlag = 0
                        # 提取当前块的位置、名字
                        PatchPosition, NameNow = GetPatchPosition(img_paths[i])
                        LastName = NameNow
                        # 创建两个全黑的三维矩阵，分别分别拼接后的预测、拼接后的Mask
                        OnePeople = np.zeros([160, 160, 160], dtype=np.uint8)
                        OneMask = np.zeros([160, 160, 160], dtype=np.uint8)
                        # 创建三个全黑的三维矩阵，分别用于预测出来的WT、TC、ET分块的拼接
                        OneWT = np.zeros([160, 160, 160], dtype=np.uint8)
                        OneTC = np.zeros([160, 160, 160], dtype=np.uint8)
                        OneET = np.zeros([160, 160, 160], dtype=np.uint8)
                        # 创建三个全黑的三维矩阵，分别用于真实的WT、TC、ET分块的拼接
                        OneWTMask = np.zeros([160, 160, 160], dtype=np.uint8)
                        OneTCMask = np.zeros([160, 160, 160], dtype=np.uint8)
                        OneETMask = np.zeros([160, 160, 160], dtype=np.uint8)
                        # 处理预测后的分块
                        # (2, 3, 32, 160, 160) output
                        # 预测分块的拼接
                        for idz in range(output.shape[2]):
                            for idx in range(output.shape[3]):
                                for idy in range(output.shape[4]):
                                    if output[i, 0, idz, idx, idy] > 0.5:  # WT拼接
                                        OneWT[PatchPosition + idz, idx, idy] = 1
                                    if output[i, 1, idz, idx, idy] > 0.5:  # TC拼接
                                        OneTC[PatchPosition + idz, idx, idy] = 1
                                    if output[i, 2, idz, idx, idy] > 0.5:  # ET拼接
                                        OneET[PatchPosition + idz, idx, idy] = 1
                        # Mask分块的拼接
                        OneWTMask[PatchPosition:(PatchPosition + output.shape[2]), :, :] = target[i, 0, :, :, :]
                        OneTCMask[PatchPosition:(PatchPosition + output.shape[2]), :, :] = target[i, 1, :, :, :]
                        OneETMask[PatchPosition:(PatchPosition + output.shape[2]), :, :] = target[i, 2, :, :, :]
                    # 提取当前块的位置、名字
                    PatchPosition, NameNow = GetPatchPosition(img_paths[i])
                    if (NameNow != LastName):
                        # 计算指标
                        CalculateWTTCET(OneWT, OneWTMask, OneTC, OneTCMask, OneET, OneETMask)
                        # OnePeople 0 1 2 4 => 增加或减少切片使得尺寸回到（155，240，240） => NII
                        for idz in range(OneWT.shape[0]):
                            for idx in range(OneWT.shape[1]):
                                for idy in range(OneWT.shape[2]):
                                    if (OneWT[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 2
                                    if (OneTC[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 1
                                    if (OneET[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 4
                        SavePeoPle = np.zeros([155, 240, 240], dtype=np.uint8)
                        SavePeoPle[:, 40:200, 40:200] = OnePeople[3:158, :, :]
                        saveout = sitk.GetImageFromArray(SavePeoPle)
                        sitk.WriteImage(saveout, savedir + LastName + ".nii.gz")

                        LastName = NameNow
                        # 创建两个全黑的三维矩阵，分别分别拼接后的预测、拼接后的Mask
                        OnePeople = np.zeros([160, 160, 160], dtype=np.uint8)
                        OneMask = np.zeros([160, 160, 160], dtype=np.uint8)
                        # 创建三个全黑的三维矩阵，分别用于预测出来的WT、TC、ET分块的拼接
                        OneWT = np.zeros([160, 160, 160], dtype=np.uint8)
                        OneTC = np.zeros([160, 160, 160], dtype=np.uint8)
                        OneET = np.zeros([160, 160, 160], dtype=np.uint8)
                        # 创建三个全黑的三维矩阵，分别用于真实的WT、TC、ET分块的拼接
                        OneWTMask = np.zeros([160, 160, 160], dtype=np.uint8)
                        OneTCMask = np.zeros([160, 160, 160], dtype=np.uint8)
                        OneETMask = np.zeros([160, 160, 160], dtype=np.uint8)
                        # 处理预测后的分块
                        # (2, 3, 32, 160, 160) output
                        # 预测分块的拼接
                        for idz in range(output.shape[2]):
                            for idx in range(output.shape[3]):
                                for idy in range(output.shape[4]):
                                    if output[i, 0, idz, idx, idy] > 0.5:  # WT拼接
                                        OneWT[PatchPosition + idz, idx, idy] = 1
                                    if output[i, 1, idz, idx, idy] > 0.5:  # TC拼接
                                        OneTC[PatchPosition + idz, idx, idy] = 1
                                    if output[i, 2, idz, idx, idy] > 0.5:  # ET拼接
                                        OneET[PatchPosition + idz, idx, idy] = 1
                        # Mask分块的拼接
                        OneWTMask[PatchPosition:(PatchPosition + output.shape[2]), :, :] = target[i, 0, :, :, :]
                        OneTCMask[PatchPosition:(PatchPosition + output.shape[2]), :, :] = target[i, 1, :, :, :]
                        OneETMask[PatchPosition:(PatchPosition + output.shape[2]), :, :] = target[i, 2, :, :, :]
                    if (NameNow == LastName):
                        # 预测分块的拼接
                        for idz in range(output.shape[2]):
                            for idx in range(output.shape[3]):
                                for idy in range(output.shape[4]):
                                    if output[i, 0, idz, idx, idy] > 0.5:  # WT拼接
                                        OneWT[PatchPosition + idz, idx, idy] = 1
                                    if output[i, 1, idz, idx, idy] > 0.5:  # TC拼接
                                        OneTC[PatchPosition + idz, idx, idy] = 1
                                    if output[i, 2, idz, idx, idy] > 0.5:  # ET拼接
                                        OneET[PatchPosition + idz, idx, idy] = 1
                        # Mask分块的拼接
                        OneWTMask[PatchPosition:(PatchPosition + output.shape[2]), :, :] = target[i, 0, :, :, :]
                        OneTCMask[PatchPosition:(PatchPosition + output.shape[2]), :, :] = target[i, 1, :, :, :]
                        OneETMask[PatchPosition:(PatchPosition + output.shape[2]), :, :] = target[i, 2, :, :, :]

                    # 最后一个分块从这里结束
                    if mynum == len(val_loader)-1:
                        # 计算指标
                        CalculateWTTCET(OneWT, OneWTMask, OneTC, OneTCMask, OneET, OneETMask)
                        # OnePeople 0 1 2 4 => 增加或减少切片使得尺寸回到（155，240，240） => NII
                        for idz in range(OneWT.shape[0]):
                            for idx in range(OneWT.shape[1]):
                                for idy in range(OneWT.shape[2]):
                                    if (OneWT[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 2
                                    if (OneTC[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 1
                                    if (OneET[idz, idx, idy] == 1):
                                        OnePeople[idz, idx, idy] = 4
                        SavePeoPle = np.zeros([155, 240, 240], dtype=np.uint8)
                        SavePeoPle[:, 40:200, 40:200] = OnePeople[3:158, :, :]
                        saveout = sitk.GetImageFromArray(SavePeoPle)
                        sitk.WriteImage(saveout, savedir + LastName + ".nii.gz")

            torch.cuda.empty_cache()
    print('WT Dice: %.4f' % np.mean(wt_dices))
    print('TC Dice: %.4f' % np.mean(tc_dices))
    print('ET Dice: %.4f' % np.mean(et_dices))
    print("=============")
    print('WT PPV: %.4f' % np.mean(wt_ppvs))
    print('TC PPV: %.4f' % np.mean(tc_ppvs))
    print('ET PPV: %.4f' % np.mean(et_ppvs))
    print("=============")
    print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
    print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
    print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
    print("=============")
    print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
    print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
    print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
    print("=============")


if __name__ == '__main__':
    main( )
