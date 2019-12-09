import matplotlib.pyplot as plt

from utils.utils import get_all_samples, draw_bbox_label
from config import cfg


if __name__ == "__main__":
    all_image_list, labels = get_all_samples(cfg.DIR_PATH)

    for i in range(len(labels)):
        image = all_image_list[i]
        bboxs, classids = labels[i]

        image = draw_bbox_label(image, bboxs, classids, 3)
        plt.imshow(image)
        plt.show()
