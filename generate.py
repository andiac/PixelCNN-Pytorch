'''
Code by Hrituraj Singh
Indian Institute of Technology Roorkee
'''

import sys
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from utils import *
from Model import PixelCNN
import pdb

# def main(config_file):
def drawsave(config_file, pathofmodel):
	config = parse_config(config_file)
	model = config['model']
	images = config['images']

	load_dir = model.get( 'load', '' )
	# load_path = model.get('load_path', 'Models/Model_Checkpoint_Last.pt')
	# load_path = model.get('load_path', pathofmodel )
	load_path = pathofmodel
	assert os.path.exists(load_path), 'Saved Model File Does not exist!'
	no_images = images.get('no_images', 144)
	images_size = images.get('images_size', 28)
	images_channels = images.get('images_channels', 1)
	

	
	#Define and load model
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = PixelCNN().to(device)
	if torch.cuda.device_count() > 1: #Accelerate testing if multiple GPUs available
  		print("Let's use", torch.cuda.device_count(), "GPUs!")
  		net = nn.DataParallel(net)
	# pdb.set_trace()
	net.load_state_dict(torch.load(load_path))
	net.eval()

	sample = torch.Tensor(no_images, images_channels, images_size, images_size).to(device)
	sample.fill_(0)

	#Generating images pixel by pixel
	for i in range(images_size):
		for j in range(images_size):
			sample_v = Variable(sample, volatile=True)
			out = net(sample_v)
			# probs = F.softmax(out[:,:,i,j], dim=-1).data
			# sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
			print(i, j, "out.shape[1]:", out.shape[1])
			out_sample = sample_from_discretized_mix_logistic_1d(out, out.shape[1] // 3)
			sample[:, :, i, j] = out_sample.data[:, :, i, j]


	#Saving images row wise
	new_dir_name = 'Models36sample_epochs250'
	if not os.path.exists(new_dir_name):
		os.mkdir( new_dir_name )
	# import pdb
	# pdb.set_trace()
	save_dir = os.path.join(new_dir_name, pathofmodel.split('/')[-1])
	torchvision.utils.save_image(sample, save_dir+'.jpg', nrow=12, padding=0)
	# torchvision.utils.save_image(sample, 'sample.png', nrow=12, padding=0)

if __name__=='__main__':
	config_file = sys.argv[1]
	assert os.path.exists(config_file), "Configuration file does not exit!"
	# main(config_file)
	model_dir = 'Models_36_epochs250'
	path_list = []
	for fname in os.listdir( model_dir ):
		# pdb.set_trace()
		# path_list.append( os.path.join( model_dir, fname ) )
		model_path = os.path.join( model_dir, fname )
		drawsave(config_file, model_path)
