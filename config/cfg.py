import os

SAVE_DIR = './weights'

SAVE_NAME = "net_"

TOTAL_EPOCH = 600

BATCH_SIZE = 7

LEARNING_RATE = 3e-4

IMAGE_DIR_PATH = "./data/modified_jpgs/JPEGImages"

DIR_PATH = "./data/modified_jpgs/JPEGImages/"

ANNOTATION_DIR_PATH = "./data/modified_jpgs/Annotations"

NUM_OF_IMAGES = len(os.listdir(IMAGE_DIR_PATH))

CLASS_NAME = ['background', 'keyPoint_1', 'keyPoint_2']

id2className_dict = {a: b for a, b in enumerate(CLASS_NAME)}

className2id_dict = {b: a for a, b in enumerate(CLASS_NAME)}
