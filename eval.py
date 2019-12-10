import torch
import torch.nn as nn
import random
import argparse
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd

from utils.utils import get_filePath_list, get_sample_image_bbox_classId, get_priorBox_2d
from utils.utils import trainDataSet, costume_collate_fn, get_device, get_all_samples
from config import cfg
from model.model import Net
from predict import detect
from sklearn.model_selection import train_test_split


def test_coord_classid(image_filePath, model):
    priorBox_2d = get_priorBox_2d()
    image_3dArray, label = get_sample_image_bbox_classId(image_filePath)
    print("Ground Truth: ", label)
    result = detect(image_3dArray, priorBox_2d, model)
    print("Result: ", result)


def eval_model(true_y, predicted_y, category_list):
    p, r, f1, s = precision_recall_fscore_support(true_y, predicted_y)
    if len(p) == len(category_list) - 1:
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
    df[column_list] = df[column_list].applymap(lambda x: '%.4f' % x)
    return df, all_f1


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
    df, f1 = eval_model(gt_y, p_y, category_list)
    return df, f1


if __name__ == "__main__":

    # all_filePath_list = get_filePath_list(cfg.DIR_PATH)
    # index = random.randint(0, len(all_filePath_list))
    # image_filePath = all_filePath_list[index]
    # test_coord_classid(image_filePath)

    # gt_y = [random.randint(0,3) for i in range(10000)]
    # p_y = [random.randint(0,3) for i in range(10000)]
    # category_list = ['background', 'keyPoint_1', 'keyPoint_2']
    # pd = eval_model(gt_y, p_y, category_list)
    # print(pd)

    device = get_device()
    parser = argparse.ArgumentParser("weight path")
    parser.add_argument('--weight_path',
                        type=str,
                        default="./weights/net_5.pth")
    args = parser.parse_args()

    model = Net()
    if args.weight_path is not "":
        model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    ######################################################################
    imageFilePath_list = get_filePath_list(cfg.IMAGE_DIR_PATH)
    index_1dArray = np.arange(cfg.NUM_OF_IMAGES)
    trainIndex_1dArray, testIndex_1dArray = train_test_split(index_1dArray,
                                                             test_size=0.2)
    allimages_list, alllabel_list = get_all_samples(cfg.DIR_PATH)
    ######################################################################

    gt_label_list = []
    p_label_list = []

    trainDatasets = trainDataSet(cfg.DIR_PATH,
                                 allimages_list,
                                 alllabel_list,
                                 trainIndex=trainIndex_1dArray)

    # trainDataLoader = DataLoader(trainDatasets,
    #                              batch_size=1,
    #                              num_workers=1,
    #                              collate_fn=costume_collate_fn,
    #                              shuffle=True)

    for index in range(len(trainDatasets)):
        image, label = trainDatasets[index]
        gt_label_list.append(label)
        # print(image.shape, '*' * 19)
        p_label = detect(image, model)
        p_label_list.append(p_label)

    pf, f1 = evals(gt_label_list, p_label_list)
    print(pf)
