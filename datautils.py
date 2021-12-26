import ast
import os
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as trn
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset
from myutils import plot_train_val, rescaling, rescaling_inv

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


def get_subset(ds, label):
    sub_indices = []
    for i in range(len(ds)):
        if ds[i][1] == label:
            sub_indices.append(i)

    return Subset(ds, sub_indices)

if __name__ == "__main__":
    label = 3

    transform_train = trn.Compose([
        trn.RandomHorizontalFlip(p=0.5),
        trn.ToTensor(),
        rescaling])

    train = datasets.CIFAR10(root='./Data',
                             train=True,
                             download=True,
                             transform=transform_train)

    sub_indices = []
    for i in range(len(train)):
        if train[i][1] == label:
            sub_indices.append(i)

    train_sub = Subset(train, sub_indices)

    for X, y in train_sub:
        print(y)
