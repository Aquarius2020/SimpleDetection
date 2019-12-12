import random
import numpy as np
import torch


t1 = torch.arange(0,60).reshape(3,4,5)

print(t1, t1.shape)

rand_2d = torch.randn(3,4) > 0

print(rand_2d, rand_2d.shape)

print(t1[rand_2d], t1[rand_2d].shape)

