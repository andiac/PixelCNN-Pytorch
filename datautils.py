import ast
import os
import torch
import torchvision
import torchvision.transforms as trn
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# y=1: indist y=0:ood
class ScTinyImagenet32(Dataset):
    def __init__(self, data_dir="./Data", imglist="./Data/test_cifar_tin.txt", 
                                          transform=trn.Compose([trn.Resize(32, interpolation=InterpolationMode.BILINEAR), 
                                                                 trn.ToTensor()])):

        self.transform = transform
        self.x = []
        self.y = []

        with open(imglist) as imgfile:
            self.imglist = imgfile.readlines()

        for line in self.imglist:
            tokens = line.strip().split(" ", 1)
            image_name, extra_str = tokens[0], tokens[1]

            # load image
            self.x.append(Image.open(os.path.join(data_dir, image_name)).convert("RGB"))

            # load label
            d = ast.literal_eval(extra_str)
            self.y.append(1. if d['sc_label'] > 0 else 0.)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.transform(self.x[idx]), self.y[idx]


if __name__ == '__main__':
    dataset = ScTinyImagenet32()

