import numpy as np
import typing

T = typing.TypeVar('T')

class Functions(object):
	@staticmethod
	# Used as an activation function for 
	# Perceptrons
	def sigmoid(x):
		return 1/(1+np.exp(-x))

	def sigmoid_der(x):
		return Functions.sigmoid(x)*(1 - Functions.sigmoid(x))
