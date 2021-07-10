# -*- coding: utf-8 -*-
import os, os.path as op

trainset_dir = '../squad_v2/trainset'
txt_tlpt = '{}_{}folds_{}.txt'


# def GetLines(txt_path):
#     return [line for line in open(txt_path).readlines() if len(line) > 0]


def WriteTxt(List, txt_path):
    f = open(txt_path, 'w+')
    for line in List:
        f.writelines(line + '\n')
    if op.exists(txt_path):
        print('Written {} lines to {}.'.format(len(List), txt_path))


def SplitFolds(data_dir, num_folds, gen_val=False):
    AllList = sorted(os.listdir(data_dir))
    unit = int(len(AllList) / num_folds)
    for i in range(1, num_folds + 1):
        j = num_folds - i
        TrainList = AllList[:unit * j] + AllList[unit * (j + 1):]
        WriteTxt(TrainList, txt_tlpt.format('train', num_folds, i))
        if gen_val:
            ValList = AllList[unit * j:unit * (j + 1)]
            WriteTxt(ValList, txt_tlpt.format('val', num_folds, i))


SplitFolds(trainset_dir, num_folds=4)
