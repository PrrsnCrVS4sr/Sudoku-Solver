# Import any ML library here (eg torch, keras, tensorflow)
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch import tensor

# Start Editing

# End Editing

import argparse
import random
import numpy as np
from dataLoader import Loader
import os
import cv2

# (Optional) If you want to define any custom module (eg a custom pytorch module), this is the place to do so
# Start Editing
# End Editing


class NeuralNetwork(nn.Module):
	def __init__(self, input_sz, hidden_sz, output_sz):
		super().__init__()
		self.linear1 = nn.Linear(input_sz, hidden_sz)
		self.linear2 = nn.Linear(hidden_sz, output_sz)

	def forward(self, x):
		out = self.linear1(x)
		out = F.relu(out)
		out = self.linear2(out)
		return out

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, stride=1,kernel_size=3,padding=0),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, stride=1,kernel_size=3,padding=0),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Dropout(0.2)
			
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, stride=1,kernel_size=3,padding=0),
			nn.ReLU(),

			
		)
		self.linear1 = nn.Linear(3*3*64,64)
		self.linear2 = nn.Linear(64,10)
		# self.linear3 = nn.Linear(64,10)
		# self.linear1 = nn.Linear(32*7*7,10)

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		# print(out.shape)
		out = self.conv3(out)
		# print(out.shape)
		out = out.view(out.size(0), -1)
		# print(out.shape)
		out = self.linear1(out)
		out = F.relu(out)
		out = self.linear2(out)
		return out

# This is the class for training our model
class Trainer:
	def __init__(self):

		# Seed the RNG's
		# This is the point where you seed your ML library, eg torch.manual_seed(12345)
		# Start Editing
		np.random.seed(12345)
		random.seed(12345)

		# End Editing

		# Set hyperparameters. Fiddle around with the hyperparameters as different ones can give you better results
		# (Optional) Figure out a way to do grid search on the hyperparameters to find the optimal set
		# Start Editing
		self.batch_size = 10  # Batch Size
		self.num_epochs = 50  # Number of Epochs to train for
		self.lr = 0.001       # Learning rate
		# End Editing

		# Init the model, loss, optimizer etc
		# This is the place where you define your model (the neural net architecture)
		# Experiment with different models
		# For beginners, I suggest a simple neural network with a hidden layer of size 32 (and an output layer of size 10 of course)
		# Don't forget the activation function after the hidden layer (I suggest sigmoid activation for beginners)
		# Also set an appropriate loss function. For beginners I suggest the Cross Entropy Loss
		# Also set an appropriate optimizer. For beginners go with gradient descent (SGD), but others can play around with Adam, AdaGrad and you can even try a scheduler for the learning rate
		# Start Editing
		self.model = CNN()
		self.loss = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.RMSprop(self.model.parameters(), self.lr)
		# End Editing

	def load_data(self):
		# Load Data
		# self.loader = Loader()

		# Change Data into representation favored by ML library (eg torch.Tensor for pytorch)
		# This is the place you can reshape your data (eg for CNN's you will want each data point as 28x28 tensor and not 784 vector)
		# Don't forget to normalize the data (eg. divide by 255 to bring the data into the range of 0-1)
		# Start Editing
		# self.train_data = torch.reshape(tensor(self.loader.train_data/255),(-1,1,28,28)).float()
		# self.test_data = torch.reshape(tensor(self.loader.test_data/255),(-1,1,28,28)).float()
		# print(self.test_data.shape)
		# self.train_data = tensor(self.loader.train_data/255)
		# self.test_data = tensor(self.loader.test_data/255)
		# self.train_labels = tensor(self.loader.train_labels)
		# self.test_labels = tensor(self.loader.test_labels)
		# self.train_data2 = np.load("data/train_data.npy")
		self.train_data = torch.reshape(tensor(np.load("data/train_data.npy")/255),(-1,1,28,28)).float()
		self.test_data = torch.reshape(tensor(np.load("data/test_data.npy")/255),(-1,1,28,28)).float()
		self.train_labels = tensor(np.load("data/train_labels.npy"))
		self.test_labels = tensor(np.load("data/test_labels.npy"))
		# print(self.train_data[0])

		# End Editing

	def save_model(self):
		# Save the model parameters into the file 'assets/model'
		torch.save(self.model.state_dict(), 'assets/model')
		# Start Editing

		# End Editing

	def load_model(self):
		# Load the model parameters from the file 'assets/model'
		if os.path.exists('assets/model'):
			self.model.load_state_dict(torch.load('assets/model'))
		else:
			raise Exception('Model not trained')

	def train(self):
		if not self.model:
			return

		print("Training...")
		for epoch in range(self.num_epochs):
			train_loss = self.run_epoch()

			# For beginners, you can leave this alone as it is
			# For others, you can try out splitting the train data into train + val data, and use the validation loss to determine whether to save the model or not
			# Start Editing
			self.save_model()
			# End Editing

			print(f'	Epoch #{epoch+1} trained')
			print(f'		Train loss: {train_loss:.3f}')
		print('Training Complete')

	def test(self):
		if not self.model:
			return 0

		print(f'Running test...')
		# Initialize running loss
		running_loss = 0.0

		# Start Editing

		# Set the ML library to freeze the parameter training

		i = 0  # Number of batches
		correct = 0  # Number of correct predictions
		for batch in range(0, self.test_data.shape[0], self.batch_size):
			# shape [batch_size,784] or [batch_size,28,28]
			batch_X =Variable(self.test_data[batch: batch+self.batch_size])
			batch_Y = Variable(self.test_labels[batch: batch +
									   self.batch_size])  # shape [batch_size,]

			# Find the predictions
			preds = self.model(batch_X)
			# Find the loss
			loss = self.loss(preds, batch_Y.long())
			# Find the number of correct predictions and update correct
			correct += preds.argmax(dim=1).eq(batch_Y.long()).sum().item()
			# Update running_loss
			running_loss += loss.item()

			i += 1

		# End Editing

		print(f'	Test loss: {(running_loss/i):.3f}')
		print(f'	Test accuracy: {(correct*100/self.test_data.shape[0]):.2f}%')

		return correct/self.test_data.shape[0]

	def run_epoch(self):
		# Initialize running loss
		running_loss = 0.0

		# Start Editing

		# Set the ML library to enable the parameter training

		# Shuffle the data (make sure to shuffle the train data in the same permutation as the train labels)

		i = 0  # Number of batches
		for batch in range(0, self.train_data.shape[0], self.batch_size):
			# shape [batch_size,784] or [batch_size,28,28]
			batch_X = Variable(self.train_data[batch: batch+self.batch_size])
			# print(batch_X.shape)
			# shape [batch_size,]
			batch_Y = Variable(self.train_labels[batch: batch+self.batch_size])

			# Zero out the grads for the optimizer
			self.optimizer.zero_grad()
			# Find the predictions
			preds = self.model(batch_X)
			# Find the loss
			loss = self.loss(preds, batch_Y.long())
			# Backpropagation
			loss.backward()

			self.optimizer.step()

			# Update the running loss
			running_loss += loss.item()
			i += 1

		# End Editing

		return running_loss / i

	def predict(self, image):
		prediction = 0
		if not self.model:
			return prediction
		# resized_image = cv2.resize(image,(28,28))
		# print(resized_image)
		img_tensored = torch.reshape(tensor(image/255),(-1,1,28,28))
		output = self.model(img_tensored.float())
		# print(F.softmax(output))
		prediction = output.argmax().item()
		# Start Editing

		# Change image into representation favored by ML library (eg torch.Tensor for pytorch)
		# img = tensor(image/255)
		# img = torch.reshape(img, (-1, 784))
		# # This is the place you can reshape your data (eg for CNN's you will want image as 28x28 tensor and not 784 vector)
		# # Don't forget to normalize the data (eg. divide by 255 to bring the data into the range of 0-1)
		# prediction = self.model(img.float()).argmax()
		# Predict the digit value using the model

		# End Editing
		return prediction


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Trainer')
	parser.add_argument('-train', action='store_true', help='Train the model')
	parser.add_argument('-test', action='store_true',
						help='Test the trained model')
	parser.add_argument('-preview', action='store_true',
						help='Show a preview of the loaded test images and their corresponding labels')
	parser.add_argument('-predict', action='store_true',
						help='Make a prediction on a randomly selected test image')

	options = parser.parse_args()

	t = Trainer()
	if options.train:
		t.load_data()
		t.train()
		t.test()
	if options.test:
		t.load_data()
		t.load_model()
		t.test()
	if options.preview:
		t.load_data()
		t.loader.preview()
	if options.predict:
		t.load_data()
		try:
			t.load_model()
		except:
			pass
		i = np.random.randint(0, t.loader.test_data.shape[0])

		print(f'Predicted: {t.predict(t.loader.test_data[i])}')
		print(f'Actual: {t.loader.test_labels[i]}')

		image = t.loader.test_data[i].reshape((28, 28))
		image = cv2.resize(image, (0, 0), fx=16, fy=16)
		cv2.imshow('Digit', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
