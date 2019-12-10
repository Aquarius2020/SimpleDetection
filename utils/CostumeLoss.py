import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    def __init__(self, priorBox_2d):
        super(MultiBoxLoss, self).__init__()
        self.priorBox_2d = priorBox_2d
        self.locationLoss = nn.MSELoss()
        self.confidenceLoss = nn.CrossEntropyLoss()

    def match(self, label_list):
        num_label = len(label_list)
        num_prior = len(self.priorBox_2d)

        gt_offset_3d = torch.zeros(num_label, num_prior, 2)
        gt_classid_2d = torch.zeros(num_label, num_prior).long()

        for index, label in enumerate(label_list):
            box_list, classid_list = label
            # print(classid_list, '='*5)
            for box, classid in zip(box_list, classid_list):
                # ground truth
                xmin, ymin, xmax, ymax = box
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                i, j = center_y // 32, center_x // 32  # 注意对应关系
                k = i * 10 + j  # 找到需要负责的priorBox
                priorBox_1d = self.priorBox_2d[k]

                gt_box_1d = torch.Tensor([center_x, center_y])
                gt_offset_1d = gt_box_1d - priorBox_1d  # 中心坐标相减，得到offset

                # gt_offset_3d用来记录，相差的offset
                gt_offset_3d[index][k] = gt_offset_1d
                gt_classid_2d[index][k] = classid
                # print(classid)

        if torch.cuda.is_available():
            gt_offset_3d = gt_offset_3d.cuda()
            gt_classid_2d = gt_classid_2d.cuda()
        # print(gt_classid_2d)
        return gt_offset_3d, gt_classid_2d

    def forward(self, prediction_3d, label_list):
        # prediction_3d : bs, w*h, channel = 2 + num_classes
        p_offset_3d = prediction_3d[:, :, :2]
        p_confidence_3d = prediction_3d[:, :, 2:]

        # gt_offset_3d是先验框与gt的偏移
        # gt_classid_2d是记录的所有id
        gt_offset_3d, gt_classid_2d = self.match(label_list)

        # 定位误差只计算正样本
        positive_2d = gt_classid_2d > 0  # 0 代表background
        # positive_2d 现在得到的是index
        positive_pOffset_2d = p_offset_3d[positive_2d]
        positive_gtOffset_2d = gt_offset_3d[positive_2d]
        location_loss = self.locationLoss(positive_gtOffset_2d,
                                          positive_pOffset_2d)

        # 误差计算1个正样本+2个负样本
        num_negtive = 2
        # p_confidence_3d: bs, w*h, num_classes
        afterSoftmaxConfidence_3d = F.softmax(p_confidence_3d, dim=2)
        # negtiveConfidence_2d ： [bs, w*h, classid=0]
        negtiveConfidence_2d = afterSoftmaxConfidence_3d[..., 0]  # class = background
        # 先按照w*h这一部分进行排序
        # negtiveConfidence_2d: [bs, w*h]
        index_2d = negtiveConfidence_2d.sort(1)[1]
        # index_2d shape：[bs, w*h]
        rank_2d = index_2d.sort(1)[1]
        # https://blog.csdn.net/LXX516/article/details/78804884

        negtive_2d = rank_2d < num_negtive
        isSelected_2d = positive_2d + negtive_2d

        p_confidence_2d = p_confidence_3d[isSelected_2d]
        gt_classid_1d = gt_classid_2d[isSelected_2d]
        # print(isSelected_2d ,p_confidence_2d, '\n', gt_classid_1d)
        confidence_loss = self.confidenceLoss(p_confidence_2d, gt_classid_1d)

        return location_loss, confidence_loss
