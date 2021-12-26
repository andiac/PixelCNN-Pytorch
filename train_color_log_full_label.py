import argparse
import torch
from torch import optim
from torch.utils import data
from torchvision import datasets, transforms

from utils import *
from Model import FullPixelCNN
from myutils import plot_train_val, rescaling, rescaling_inv
from datautils import get_subset
from mylogger import MyLogger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--label", type=int, default=0, help="label")
    args = parser.parse_args()

    batch_size = 100
    lr         = 3e-4
    epochs     = 550
    save_path  = f"./Model/color_log_full_{args.label}" 
    logger = MyLogger(f"color_log_full_{args.label}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        rescaling])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        rescaling])

    train = get_subset(datasets.CIFAR10(root='./Data',
                             train=True,
                             download=True,
                             transform=transform_train), args.label)

    val   = get_subset(datasets.CIFAR10(root='./Data',
                             train=False,
                             download=True,
                             transform=transform_val), args.label)

    N_train = len(train)
    N_val= len(val)

    train = data.DataLoader(train, batch_size=batch_size, shuffle=True , num_workers=1, pin_memory=True)
    val   = data.DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = FullPixelCNN(res_num=10, in_channels=3, out_channels=100).to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_losses = []
    val_losses   = []
    for epoch in range(epochs):
        net.train()
        train_loss_sum = 0.0
        
        for images, labels in train:
            images = images.to(device)
            optimizer.zero_grad()

            outputs = net(images)
            loss = discretized_mix_logistic_loss(images, outputs)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        net.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, labels in val:
                images = images.to(device)
                
                outputs = net(images)
                loss = discretized_mix_logistic_loss(images, outputs)
                val_loss_sum += loss.item()

        train_loss_mean = train_loss_sum / N_train
        val_loss_mean = val_loss_sum / N_val
        train_losses.append(train_loss_mean)
        val_losses.append(val_loss_mean)

        logger.info(f"Epoch: {epoch}, train loss: {train_loss_mean}, val loss: {val_loss_mean}")
        if epoch > 100 and val_loss_mean < min(val_losses[:-1]):
            torch.save(net.state_dict(), os.path.join(save_path, f"checkpoint_{epoch}.pt"))

    plot_train_val(train_losses, val_losses, f"color_log_full_{args.label}.png")

