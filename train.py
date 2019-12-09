import argparse
import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import cfg
from dataloader import testIndex_1dArray, trainIndex_1dArray
from eval import evals
from model.model import Net
from predict import detect
from utils.CostumeLoss import MultiBoxLoss
from utils.utils import (costume_collate_fn, get_all_samples, get_device,
                         get_filePath_list, get_priorBox_2d, print_flush,
                         testDataSet, trainDataSet)
from utils.visdom import Visualizer


def train(total_epoch, model, optimizer, scheduler, criterion,
          trainDataLoader, testDatasets):

    vis = Visualizer(env="detection")

    model.train()
    for epoch in range(total_epoch):
        for index, (batch_image, label_list) in enumerate(trainDataLoader):
            if torch.cuda.is_available():
                batch_image = batch_image.cuda()

            prediction_3d = model(batch_image)
            # print(batch_image.shape, label_list)
            location_loss, confidence_loss = criterion(prediction_3d,
                                                       label_list)
            loss = location_loss + confidence_loss * 20

            loss_value = loss.item()
            location_loss_value = location_loss.item()
            confidence_loss_value = confidence_loss.item()

            print_string = "epoch:%d | batch:%04d | loc_loss:%.5f | conf_loss:%.5f | loss:%.5f | lr:%.6f" % (
                epoch, index, location_loss_value, confidence_loss_value * 20,
                loss_value, scheduler.get_lr()[0])
            vis.plot_many_stack({
                "loc_loss": location_loss_value,
                "conf_loss": confidence_loss_value * 20,
                "loss": loss_value
            })

            print(print_string)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(cfg.SAVE_DIR,
                             cfg.SAVE_NAME + str(epoch) + ".pth"))
        # test
        F1_score = test(testDatasets, model)
        print_test_string = "Epoch:%d | F1-Score: %.5f" % (epoch, F1_score)
        print(print_test_string)
        vis.plot_many_stack({"f1 score": F1_score})

        scheduler.step()


def test(testDatasets, model):
    gt_label_list = []
    p_label_list = []
    for index in range(len(testDatasets)):
        image, label = testDatasets[index]
        gt_label_list.append(label)
        # print(image.shape, '*' * 19)
        p_label = detect(image, model)
        p_label_list.append(p_label)
    pf, f1 = evals(gt_label_list, p_label_list)
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("weight path")
    parser.add_argument('--weight_path', type=str, default="", help="weight path")
    args = parser.parse_args()

    device = get_device()

    total_epoch = cfg.TOTAL_EPOCH

    dir_path = cfg.DIR_PATH

    model = Net()
    if args.weight_path is not "":
        model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    
    num_of_train = len(trainIndex_1dArray)

    batch_size = cfg.BATCH_SIZE

    epoch_size = math.ceil(num_of_train / batch_size)

    ######################################################################
    imageFilePath_list = get_filePath_list(cfg.IMAGE_DIR_PATH)
    index_1dArray = np.arange(cfg.NUM_OF_IMAGES)
    trainIndex_1dArray, testIndex_1dArray = train_test_split(index_1dArray,
                                                             test_size=0.2)
    allimages_list, alllabel_list = get_all_samples(cfg.DIR_PATH)
    ######################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    milestone_list = [10 * k for k in range(1, total_epoch // 10)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestone_list,
                                                     gamma=0.5)

    trainDatasets = trainDataSet(dir_path,
                                 allimages_list,
                                 alllabel_list,
                                 trainIndex=trainIndex_1dArray)
    trainDataLoader = DataLoader(trainDatasets,
                                 shuffle=False,
                                 collate_fn=costume_collate_fn,
                                 batch_size=cfg.BATCH_SIZE,
                                 num_workers=8)
    testDatasets = testDataSet(dir_path,
                               allimages_list,
                               alllabel_list,
                               testIndex=testIndex_1dArray)


    priorBox_2d = get_priorBox_2d()

    criterion = MultiBoxLoss(priorBox_2d)

    train(total_epoch, model, optimizer, scheduler, criterion, trainDataLoader, testDatasets)
