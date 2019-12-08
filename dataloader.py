from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Dataset

from utils.utils import get_filePath_list
from utils.utils import get_bbox_classId, get_all_samples
from config import cfg

imageFilePath_list = get_filePath_list(cfg.IMAGE_DIR_PATH)

index_1dArray = np.arange(cfg.NUM_OF_IMAGES)

trainIndex_1dArray, testIndex_1dArray = train_test_split(index_1dArray,
                                                         test_size=0.2)

if __name__ == "__main__":
    # print(imageFilePath_list)
    # print(index_1dArray)
    # print(len(trainIndex_1dArray), len(testIndex_1dArray))
    dirPath = "./data/modified_jpgs/JPEGImages/"
    trainDataSets = trainDataSet(dirPath)
    train_dataloader = DataLoader(trainDataSets, num_workers=1, batch_size=2)

    for image, bbox, classid in train_dataloader:
        print(image.shape, bbox, classid)