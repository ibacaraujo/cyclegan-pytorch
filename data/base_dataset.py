import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch.transforms as transforms
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = self.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass

    def get_params(opt, size):
        w, h = size
        new_h = h
        new_w = w
        if opt.preprocess == 'resize_and_crop':
            new_h = new_w = opt.load_size
        if opt.preprocess == 'scale_witdh_and_crop':
            new_w = opt.load_size
            new_h = opt.load_size * h // w

        x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}
