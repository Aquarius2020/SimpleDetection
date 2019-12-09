import torch
import torch.nn as nn
import random
import argparse
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd

from dataloader import testIndex_1dArray
from utils.utils import get_filePath_list, get_sample_image_bbox_classId
from utils.utils import trainDataSet, costume_collate_fn
from config import cfg
from model.model import Net
from predict import detect

def test_coord_classid(image_filePath):
    image_3dArray, label = get_sample_image_bbox_classId(image_filePath)
    print("Ground Truth: ", label)
    result = detect(image_3dArray)
    print("Result: ", result)

def eval_model(true_y, predicted_y, category_list):
    p, r, f1, s = precision_recall_fscore_support(true_y, predicted_y)
    if len(p) == len(category_list) -1:
        # 最极端的情况: 所有测试样例都正确，即没有负样本
        category_list = category_list[1:]
    category_1dArray = np.array(category_list)
    df = pd.DataFrame([category_1dArray, p, r, f1, s]).T
    df.columns = ['Label', 'Precision', 'Recall', 'F1', 'Support']
    # 计算总体的平均Precision, Recall, F1, Support
    all_label = 'Total'
    all_p = np.average(p, weights=s)
    all_r = np.average(r, weights=s)
    all_f1 = np.average(f1, weights=s)
    all_s = np.sum(s)
    row = [all_label, all_p, all_r, all_f1, all_s]
    df.loc[3] = row
    # 设置Precision、Recall、F1这3列显示4位小数
    column_list = ['Precision', 'Recall', 'F1']
    df[column_list] = df[column_list].applymap(lambda x: '%.4f' %x)
    return df


def evals(gtLabel_list, pLabel_list):
    gt_y = []
    p_y = []
    for gtLabel, pLabel in zip(gtLabel_list, pLabel_list):
        gtBox_list, gtClassId_list = gtLabel
        pBox_list, pClassId_list = pLabel
        pMatched_list = [False] * len(pBox_list)
        # 先遍历真实值
        for gtBox, gtClassId in zip(gtBox_list, gtClassId_list):
            if gtClassId in pClassId_list:
                index = pClassId_list.index(gtClassId)
                pBox = pBox_list[index]
                diffValue_1dArray = np.subtract(gtBox, pBox)
                absValue_1dArray = np.abs(diffValue_1dArray)
                diffSum = absValue_1dArray.sum()
                if diffSum < 20:
                    gt_y.append(gtClassId)
                    p_y.append(gtClassId)
                    pMatched_list[index] = True
                    continue
            gt_y.append(gtClassId)
            p_y.append(0)
        # 然后遍历预测值中未被匹配到的, 即背景被预测为正样本
        for index, matched in enumerate(pMatched_list):
            if not matched:
                pClassId = pClassId_list[index]
                gt_y.append(0)
                p_y.append(pClassId)
    category_list = ['background', 'keyPoint_1', 'keyPoint_2']
    df = eval_model(gt_y, p_y, category_list)
    return df



if __name__ == "__main__":

    all_filePath_list = get_filePath_list(cfg.DIR_PATH)
    # index = random.randint(0, len(all_filePath_list))
    # image_filePath = all_filePath_list[index]
    # test_coord_classid(image_filePath)

    # gt_y = [random.randint(0,3) for i in range(10000)]
    # p_y = [random.randint(0,3) for i in range(10000)]
    # category_list = ['background', 'keyPoint_1', 'keyPoint_2']
    # pd = eval_model(gt_y, p_y, category_list)
    # print(pd)

    gt_label_list = []
    p_label_list = []

    trainDatasets = trainDataSet(cfg.DIR_PATH)
    trainDataLoader = DataLoader(trainDatasets,
                                batch_size=1,
                                num_workers=1,
                                collate_fn=costume_collate_fn,
                                shuffle=True)
    for image_3dArray, label_list in trainDataLoader:
        gt_label_list.append(label_list)
        p_label = detect(image_3dArray)
        p_label_list.append(p_label)

    pf = evals(gtLabel_list, pLabel_list)
    print(pf)
