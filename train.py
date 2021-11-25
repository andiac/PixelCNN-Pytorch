'''
Code by Hrituraj Singh
Indian Institute of Technology Roorkee
'''

import sys
import os
import time
import torch
from torch import optim
from torch.utils import data
from torch.autograd import Variable
import torch.nn as nn
from utils import *
from Model import PixelCNN
import matplotlib.pyplot as plt 


def main(config_file):
	config = parse_config(config_file)
	data_ = config['data']
	network = config['network']

	path = data_.get('path', 'Data') #Path where the data after loading is to be saved
	data_name = data_.get('data_name','MNIST') #What data type is to be loaded ex - MNIST, CIFAR
	batch_size = data_.get('batch_size', 144)

	layers = network.get('no_layers', 8) #Number of layers in the network
	kernel = network.get('kernel', 7) #Kernel size
	channels = network.get('channels', 64) #Depth of the intermediate layers
	epochs = network.get('epochs', 25) #No of epochs
	save_path = network.get('save_path', 'Models_36_epochs250') #path where the models are to be saved


	#Loading Data
	if (data_name=='MNIST'):
		train, test = get_MNIST(path)

	train = data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers =1, pin_memory = True)
	test = data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers =1, pin_memory = True)


	#Defining the model and training it on loss function
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = PixelCNN().to(device)
	if torch.cuda.device_count() > 1: # If more than one GPU available, accelerate the training using multiple GPUs
  		print("Let's use", torch.cuda.device_count(), "GPUs!")
  		net = nn.DataParallel(net)
	
	optimizer = optim.Adam(net.parameters())
	criterion = nn.CrossEntropyLoss()
	total_train_loss = []
	total_test_loss = []

	time_start = time.time()
	print('Training Started')

	for i in range(epochs):

		
		net.train(True)
		step = 0
		train_loss_cal = 0

		for images, labels in train:
			
			target = Variable(images[:,0,:,:]*255).long()
			images = images.to(device)
			target = target.to(device)
			
			optimizer.zero_grad()

			output = net(images)
			# loss = criterion(output, target)
			# print(images.shape)
			# print(output.shape)
			loss = discretized_mix_logistic_loss_1d (images, output)
			loss.backward()
			optimizer.step()

			train_loss_cal += loss.item()/batch_size

			step+=1

			# if(step%100 == 0):
				# print('Epoch:'+str(i)+'\t'+ str(step) +'\t Iterations Complete \t'+'loss: ', loss.item()/1000.0)
				# print('Epoch:'+str(i)+'\t'+ str(step) +'\t Iterations Complete \t'+'loss: ', loss.item()/1000.0)
		
		total_train_loss.append( train_loss_cal )
		print('Epoch:'+str(i)+'\t'+ str(step) +'\t Iterations Complete \t'+'train_loss: ', train_loss_cal)
		# print('Epoch: '+str(i)+' Over!')


		net.eval()
		test_loss = 0.
		for images, labels in test:
			target = Variable(images[:,0,:,:]*255).long()
			images = images.to(device)
			target = target.to(device)

			output = net(images)
			loss = discretized_mix_logistic_loss_1d(images, output)
			test_loss += loss.item()/batch_size
		total_test_loss.append(test_loss)

		print('Epoch:'+str(i)+'\t'+ str(step) +'\t Iterations Complete \t'+'test_loss: ', test_loss)
		print('Epoch: '+str(i)+' Over!')


		#Saving the model
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		print("Saving Checkpoint!")
		if(i==epochs-1):
			torch.save(net.state_dict(), save_path+'/Model_Checkpoint_'+'Last'+'.pt')
		else:
			torch.save(net.state_dict(), save_path+'/Model_Checkpoint_'+str(i)+'.pt')
		print('Checkpoint Saved')

	print('Training Finished! Time Taken: ', time.time() - time_start)
	# import pdb
	# pdb.set_trace()
	x1 = range(0, epochs)
	x2 = range(0, epochs)
	y1 = total_train_loss
	y2 = total_test_loss
	plt.subplot(2,1,1)
	plt.plot(x1, y1, 'o')
	plt.title('LOSS')
	plt.ylabel('Train loss')
	plt.subplot(2,1,2)
	plt.plot(x2, y2, 'o')
	plt.xlabel('Epochs')
	plt.ylabel('Test loss')
	# plt.show()
	plt.savefig('loss_fig_250.jpg')

if __name__=="__main__":
	config_file = sys.argv[1]
	assert os.path.exists(config_file), "Configuration file does not exit!"
	main(config_file)




