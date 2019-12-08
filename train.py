import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math

from config import cfg
from utils.utils import get_device
from model.model import Net
from dataloader import testIndex_1dArray, trainIndex_1dArray
from utils.utils import trainDataSet, costume_collate_fn, get_priorBox_2d, print_flush
from utils.CostumeLoss import MultiBoxLoss


def train(total_epoch, model, optimizer, scheduler, criterion,
          trainDataLoader):

    model.train()
    for epoch in range(total_epoch):
        for index, (batch_image, label_list) in enumerate(trainDataLoader):
            prediction_3d = model(batch_image)
            # print(batch_image.shape, label_list)
            location_loss, confidence_loss = criterion(prediction_3d,
                                                       label_list)
            loss = location_loss + confidence_loss * 20

            loss_value = loss.item()
            location_loss_value = location_loss.item()
            confidence_loss_value = confidence_loss.item()

            print_string = "epoch:%d|batch:%04d|loc_loss:%.5f|conf_loss:%.5f|loss:%.5f" % (
                epoch, index, location_loss_value, confidence_loss_value * 20,
                loss_value)

            print(print_string)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()


if __name__ == "__main__":
    device = get_device()

    total_epoch = cfg.TOTAL_EPOCH

    dir_path = cfg.DIR_PATH

    model = Net()
    model = model.to(device)

    num_of_train = len(trainIndex_1dArray)

    batch_size = cfg.BATCH_SIZE

    epoch_size = math.ceil(num_of_train / batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    milestone_list = [10 * k for k in range(1, total_epoch // 10)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestone_list,
                                                     gamma=0.5)

    trainDataSets = trainDataSet(dir_path)
    trainDataLoader = DataLoader(trainDataSets,
                                 shuffle=False,
                                 collate_fn=costume_collate_fn,
                                 batch_size=cfg.BATCH_SIZE,
                                 num_workers=1)

    priorBox_2d = get_priorBox_2d()

    criterion = MultiBoxLoss(priorBox_2d)

    train(total_epoch, model, optimizer, scheduler, criterion, trainDataLoader)
