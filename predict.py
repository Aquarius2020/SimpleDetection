import torch
import numpy as np
import torch.nn.functional as F
import argparse
import cv2
import matplotlib.pyplot as plt
import os

from utils.utils import get_device, get_priorBox_2d, draw_bbox_label
from model.model import Net

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def detect(image_3dArray, model):
    priorBox_2d = get_priorBox_2d()
    image_4dArray = np.expand_dims(np.array(image_3dArray), 0)
    with torch.no_grad():
        if torch.cuda.is_available():
            x = torch.ByteTensor(image_4dArray).cuda()
        else:
            x = torch.ByteTensor(image_4dArray)
        x = x.permute(0, 3, 1, 2).float()
        # bs, w*h, 2+num_of_classes
        prediction_3d = model(x)
        # print(prediction_3d.shape)
    # prediction 2d: batch size, w*h, c = 2(offset)+num_classes
    prediction_2d = prediction_3d[0]
    p_offset_2d = prediction_2d[:, :2]
    p_confidence_2d = F.softmax(prediction_2d[:, 2:], 1)

    p_bbox_list = []
    p_classid_list = []

    for classid in [1, 2]:
        class_p_conf_1d = p_confidence_2d[:, classid]
        max_class_p_conf = class_p_conf_1d.max().item()
        if max_class_p_conf > 0.5:
            # append class id
            p_classid_list.append(classid)
            # 获取confidence对应的下角标以便寻找bounding box
            index = class_p_conf_1d.argmax()
            p_offset = p_offset_2d[index]
            p_center = priorBox_2d[index] + p_offset.to('cpu')
            center_x, center_y = p_center
            center_x, center_y = int(center_x), int(center_y)

            min_x = max(center_x - 15, 0)
            max_x = min(center_x + 15, 320 - 1)
            min_y = max(center_y - 15, 0)
            max_y = min(center_y + 15, 1920 - 1)

            bbox = min_x, min_y, max_x, max_y
            # append bbox
            p_bbox_list.append(bbox)
    return p_bbox_list, p_classid_list


if __name__ == "__main__":
    paser = argparse.ArgumentParser("parser")
    paser.add_argument('--image_dir',
                       type=str,
                       default="./data/modified_jpgs/JPEGImages",
                       help="single image path")
    paser.add_argument('--weight_path',
                       type=str,
                       default="./data/trained_weights/ckpt.pth",
                       help="weight path")

    args = paser.parse_args()

    device = get_device()
    model = Net()
    if args.weight_path is not "":
        model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    for item in os.listdir(args.image_dir):
        if item.endswith('jpg'):
            full_path = os.path.join(args.image_dir, item)
            image = cv2.imread(full_path)
            p_bbox_list, p_classid_list = detect(image, model)
            line_width = 2
            print(p_bbox_list)
            if len(p_bbox_list) > 0:
                image = draw_bbox_label(image, p_bbox_list, p_classid_list,
                                        line_width)
                plt.imshow(image)
                # plt.show()
