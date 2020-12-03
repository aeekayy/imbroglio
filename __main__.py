#!/usr/bin/env python3

from imbroglio.lib import Functions
import numpy as np

def main():
	# Define input features:
	print("Adding inputs")
	print("Array format: [Core CPU, Media CPU, SLA")
	input_features = np.array([[0.1, 0.12, 0.99942], [0.7, 0.24, 0.99999], [0.21, 0.24, 0.90442], [0.13, 0.13, 0.999758], [0.08, 0.084, 0.9999992], [0.31, 0.42, 0.999442]])
	#input_features = np.array([[0,0,0],[1,0,0],[0,0,1],[0,0,0],[0,0,0],[0,0,0]])
	print(input_features.shape)
	print(input_features)
	
	# Define target output :
	print("Adding target outputs")
	target_output = np.array([[0,1,1,0,0,0]])
	# Reshaping our target output into vector :
	target_output = target_output.reshape(6,1)
	print(target_output.shape)
	print (target_output)
	
	# Define weights :
	print("Adding weights")
	weights = np.array([[0.25],[0.25],[0.5]])
	print(weights.shape)
	print (weights)
	
	# Define learning rate :
	print("Define learning rate")
	lr = 0.05
	
	# Main logic for neural network :
	# Running our code 10000 times :
	for epoch in range(10000):
		inputs = input_features
		#Feedforward input :
		pred_in = np.dot(inputs, weights)
		#Feedforward output :
		pred_out = Functions.sigmoid(pred_in)
		#Backpropogation 
		#Calculating error
		error = pred_out - target_output
		x = error.sum()
	
		#Going with the formula :
		print(x)
	
		#Calculating derivative :
		dcost_dpred = error
		dpred_dz = Functions.sigmoid_der(pred_out)

		#Multiplying individual derivatives :
		z_delta = dcost_dpred * dpred_dz
		#Multiplying with the 3rd individual derivative :
		inputs = input_features.T
		weights -= lr * np.dot(inputs, z_delta)
	
	print("Final weights are: ")
	print(weights)
	#Taking inputs :
	single_point = np.array([0.3,0.3,0.27])
	#1st step :
	result1 = np.dot(single_point, weights)
	#2nd step :
	result2 = Functions.sigmoid(result1)
	#Print final result
	print(result2)
	#Taking inputs :
	single_point = np.array([0.2,0.2,0.9939])
	#1st step :
	result1 = np.dot(single_point, weights)
	#2nd step :
	result2 = Functions.sigmoid(result1)
	#Print final result
	print(result2)
	#Taking inputs :
	single_point = np.array([.7,.1,0.9999])
	#1st step :
	result1 = np.dot(single_point, weights)
	#2nd step :
	result2 = Functions.sigmoid(result1)
	#Print final result
	print(result2)	

if __name__ == '__main__':
	main()
