import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from tqdm import tqdm

from Model import FullPixelCNN
from utils import sample_from_discretized_mix_logistic

def generate(pt_path, png_path):
    num_images = 144
    image_size = 32
    image_channel = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = FullPixelCNN(res_num=5, in_channels=3, num_out=100).to(device)

    net.load_state_dict(torch.load(pt_path))
    net.eval()

    sample = torch.Tensor(num_images, image_channel, image_size, image_size).to(device)
    sample.fill_(0)

    # Generating images pixel by pixel
    with torch.no_grad():
        for i in range(image_size):
            for j in range(image_size):
                out = net(sample)
                out_sample = sample_from_discretized_mix_logistic(out, out.shape[1] // 10)
                sample[:, :, i, j] = out_sample.data[:, :, i, j]

    torchvision.utils.save_image(sample, png_path, nrow=12, padding=0)


if __name__ == '__main__':
    step = 5
    for i in tqdm(range(0, 500, step), desc="Generating..."):
        pt_path  = f"./Model/color_log_full/checkpoint_{i}.pt"
        png_path = f"./Samples/color_log_full/checkpoint_{i}.png"
        png_folder = os.path.split(png_path)[0]
        if not os.path.exists(png_folder):
            os.makedirs(png_folder)
        generate(pt_path, png_path)

