#!/usr/bin/env python3

from imbroglio.lib import Functions
import numpy as np

def main():
	print(Functions.sigmoid(0))
	print(Functions.sigmoid_der(1))

if __name__ == '__main__':
	main()
