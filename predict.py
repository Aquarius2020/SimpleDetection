import torch
import numpy as np
import torch.nn.functional as F

from utils.utils import get_device, get_priorBox_2d
from model.model import Net


device = get_device()
model = Net()
priorBox_2d = get_priorBox_2d()

def detect(image):
    image_4dArray = np.expand_dims(np.array(image), axis=0)
    with torch.no_grad():
        x = torch.ByteTensor(image_4dArray).to(device)
        x = x.permute(0,3,1,2).float()
        prediction_3d = model(x)
    # prediction 2d: batch size, w*h, c = 2(offset)+num_classes
    prediction_2d = prediction_3d[0]
    p_offset_2d = prediction_2d[:,:,:2]
    p_confidence_2d = F.softmax(prediction_2d[:, :, 2:],1)

    p_bbox_list = []
    p_classid_list = []

    for classid in [1,2]:
        class_p_conf_1d = p_confidence_1d[:,classid]
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
    return p_classid_list, p_bbox_list
