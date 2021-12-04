'''
Code by Hrituraj Singh
Indian Institute of Technology Roorkee
'''


import torch
from MaskedCNN import MaskedCNN
import torch.nn as nn

class FullPixelCNN(nn.Module):
    """
    Network of PixelCNN as described in A Oord et. al.
    """
    def __init__(self, res_num=10, in_kernel = 7,  in_channels=3, channels=256, out_channels=256, device=None):
        super(FullPixelCNN, self).__init__()
        self.channels = channels
        self.layers = {}
        self.device = device
        self.res_num=res_num


        self.in_cnn=MaskedCNN('A',in_channels,channels, in_kernel, 1, in_kernel//2, bias=False)
        self.activation=nn.ReLU()

        self.resnet_cnn11=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn3=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 3, 1, 1) for i in range(0,res_num)])
        self.resnet_cnn12=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])

        self.out_cnn1=nn.Conv2d(channels, channels, 1)
        self.out_cnn2=nn.Conv2d(channels, out_channels, 1)


    def forward(self, x):
        x=self.in_cnn(x)
        x=self.activation(x)

        for i in range(0, self.res_num):
            x_mid=self.resnet_cnn11[i](x)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn3[i](x_mid)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn12[i](x_mid)
            x_mid=self.activation(x_mid)
            x=x+x_mid
        x=self.out_cnn1(x)
        x=self.activation(x)
        x=self.out_cnn2(x)
        return x


class LocalPixelCNN(nn.Module):
    """
    Network of PixelCNN as described in A Oord et. al.
    """
    def __init__(self, res_num=10, in_kernel = 7,  in_channels=3, channels=256, out_channels=256, device=None):
        super(LocalPixelCNN, self).__init__()
        self.channels = channels
        self.layers = {}
        self.device = device
        self.res_num=res_num


        self.in_cnn=MaskedCNN('A',in_channels,channels, in_kernel, 1, in_kernel//2, bias=False)
        self.activation=nn.ReLU()

        self.resnet_cnn11=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn3=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn12=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])

        self.out_cnn1=nn.Conv2d(channels, channels, 1)
        self.out_cnn2=nn.Conv2d(channels, out_channels, 1)


    def forward(self, x):
        x=self.in_cnn(x)
        x=self.activation(x)

        for i in range(0, self.res_num):
            x_mid=self.resnet_cnn11[i](x)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn3[i](x_mid)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn12[i](x_mid)
            x_mid=self.activation(x_mid)
            x=x+x_mid
        x=self.out_cnn1(x)
        x=self.activation(x)
        x=self.out_cnn2(x)
        return x


class PixelCNN(nn.Module):
	"""
	Network of PixelCNN as described in A Oord et. al. 
	"""
	def __init__(self, no_layers=8, kernel = 7, channels=64, in_channels=1, num_out=36, device=None):
		super(PixelCNN, self).__init__()
		self.no_layers = no_layers
		self.kernel = kernel
		self.channels = channels
		self.layers = {}
		self.device = device

		self.Conv2d_1 = MaskedCNN('A', in_channels, channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_1 = nn.BatchNorm2d(channels)
		self.ReLU_1= nn.ReLU(True)

		self.Conv2d_2 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_2 = nn.BatchNorm2d(channels)
		self.ReLU_2= nn.ReLU(True)

		self.Conv2d_3 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_3 = nn.BatchNorm2d(channels)
		self.ReLU_3= nn.ReLU(True)

		self.Conv2d_4 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_4 = nn.BatchNorm2d(channels)
		self.ReLU_4= nn.ReLU(True)

		self.Conv2d_5 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_5 = nn.BatchNorm2d(channels)
		self.ReLU_5= nn.ReLU(True)

		self.Conv2d_6 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_6 = nn.BatchNorm2d(channels)
		self.ReLU_6= nn.ReLU(True)

		self.Conv2d_7 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_7 = nn.BatchNorm2d(channels)
		self.ReLU_7= nn.ReLU(True)

		self.Conv2d_8 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_8 = nn.BatchNorm2d(channels)
		self.ReLU_8= nn.ReLU(True)

		self.out = nn.Conv2d(channels, num_out, 1)

	def forward(self, x):
		x = self.Conv2d_1(x)
		x = self.BatchNorm2d_1(x)
		x = self.ReLU_1(x)

		x = self.Conv2d_2(x)
		x = self.BatchNorm2d_2(x)
		x = self.ReLU_2(x)

		x = self.Conv2d_3(x)
		x = self.BatchNorm2d_3(x)
		x = self.ReLU_3(x)

		x = self.Conv2d_4(x)
		x = self.BatchNorm2d_4(x)
		x = self.ReLU_4(x)

		x = self.Conv2d_5(x)
		x = self.BatchNorm2d_5(x)
		x = self.ReLU_5(x)

		x = self.Conv2d_6(x)
		x = self.BatchNorm2d_6(x)
		x = self.ReLU_6(x)

		x = self.Conv2d_7(x)
		x = self.BatchNorm2d_7(x)
		x = self.ReLU_7(x)

		x = self.Conv2d_8(x)
		x = self.BatchNorm2d_8(x)
		x = self.ReLU_8(x)

		return self.out(x)







