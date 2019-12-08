import os
import random
import sys
from xml.etree import ElementTree as ET
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

main_path = os.path.dirname(os.path.join(os.getcwd(), ".."))
sys.path.append(main_path)

from config import cfg

def get_device(force_cpu=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    if cuda:
        print("Using CUDA")
    else:
        print("Using CPU")

    return device

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_bbox_classId(xmlFilePath):
    """[解析一个xml文件，得到一张图中所有的框和对应的id(类别对应的的id)]
    """
    if not os.path.exists(xmlFilePath):
        return []
    with open(xmlFilePath, encoding='utf8') as f:
        fileContent = f.read()
    root = ET.XML(fileContent)
    object_list = root.findall('object')
    classId_list = []
    bbox_list = []
    for object_item in object_list:
        # get id
        className = object_item.find('name').text
        classId = cfg.className2id_dict[className]
        classId_list.append(classId)

        # get bbox
        bbox = object_item.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)

        box_list = [xmin, ymin, xmax, ymax]
        bbox_list.append(box_list)
    label = [bbox_list, classId_list]
    return label


def get_sample_image_bbox_classId(jpgFilePath):
    """通过图像地址，解析得到标注文件地址，然后最终得到image, bbox, classid
    """
    image = cv2.imread(jpgFilePath)
    img_3dArray = np.array(image)

    baseName = os.path.basename(jpgFilePath)
    xmlName = baseName.replace("jpg", "xml")
    xmlFilePath = os.path.join(cfg.ANNOTATION_DIR_PATH, xmlName)

    label = get_bbox_classId(xmlFilePath)
    return img_3dArray, label


def print_flush(str):
    print(str, end="\r")
    sys.stdout.flush()


def get_filePath_list(dirPath):
    all_filePath_list = [
        os.path.join(dirPath, item) for item in os.listdir(dirPath)
    ]
    return all_filePath_list


def get_all_samples(dirPath):
    image_4dArray = None
    label_list = []

    all_filePath_list = get_filePath_list(dirPath)
    for item in all_filePath_list:
        img_list, label = get_sample_image_bbox_classId(item)
        img = img_list[np.newaxis, :]
        if image_4dArray is None:
            image_4dArray = img
        else:
            image_4dArray = np.concatenate((image_4dArray, img))
        label_list.append(label)
    return image_4dArray, label_list


class trainDataSet(Dataset):
    def __init__(self, dirPath, transforms=None):
        super(trainDataSet, self).__init__()
        allimages_list, alllabel_list = get_all_samples(dirPath)
        self.allimages_list = allimages_list
        self.alllabel_list = alllabel_list
        self.transforms = transforms

    def __len__(self):
        return len(self.alllabel_list)

    def __getitem__(self, index):
        image = self.allimages_list[index]
        label = self.alllabel_list[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

def costume_collate_fn(batch):
    img, label = zip(*batch)
    img = torch.Tensor(img)
    # img: bs, w, h, c
    img = img.permute(0,3,1,2)
    return img, label


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {
            key: default_collate([d[key] for d in batch])
            for key in batch[0]
        }
    elif isinstance(batch[0], tuple) and hasattr(batch[0],
                                                 '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples)
                                for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))


def get_priorBox_2d():
    box_list = []
    for i in range(60):
        for j in range(10):
            x = (j+0.5)*32
            y = (i+0.5)*32
            box = x,y
            box_list.append(box)
    priorBox_2d = torch.Tensor(box_list)
    return priorBox_2d

if __name__ == "__main__":
    dirPath = "./data/modified_jpgs/JPEGImages/"
    trainDataSets = trainDataSet(dirPath)
    # print(
    #     type(trainDataSets[0]), isinstance(trainDataSets[0], torch.Tensor),
    #     isinstance(trainDataSets[0], tuple)
    #     , hasattr(trainDataSets[0], '_fields'))
    train_dataloader = DataLoader(trainDataSets,
                                  num_workers=1,
                                  batch_size=3,
                                  shuffle=False,
                                  collate_fn=costume_collate_fn)

    for image, label in train_dataloader:
        print(image.shape, label, '\n')


    # for i in range(len(trainDataSets)):
    #     image, bbox, classid = trainDataSets[i]
    #     print(image.shape, bbox[0], classid)

    # for i in range(10000):
    #     print_string(str(i))
    # images, bboxs, classids = get_all_samples(dirPath)
    # print(images.shape,len(bboxs), len(classids))
    # print(bboxs)
    # print(classids)

    # for i in os.listdir(dirPath):
    #     imagePath = os.path.join(dirPath, i)
    #     img, bbox, label = get_sample_image_bbox_classId(imagePath)
    #     print(img.shape, len(bbox), len(label), bbox)
    # print(get_filePath_list(dirPath))
    # print(get_filePath_list2(dirPath))

    # img, bbox, label = get_all_samples(dirPath)
    # for i in range(121):
    #     print(img[i].shape, (bbox[i]), (label[i]))

    # box_list = get_priorBox_2d()
    # print(box_list.shape)
    # print(box_list[:5])
