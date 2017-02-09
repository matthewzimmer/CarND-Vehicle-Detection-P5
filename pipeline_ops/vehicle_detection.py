import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pipeline_ops.pipeline_ops import PipelineOp


class CarsNotCarsDatasetOp(PipelineOp):
	"""
	Returns a dictionary with a 'cars' array and a 'not_cars' array.
	"""

	def __init__(self):
		PipelineOp.__init__(self)
		self.__cars = []
		self.__not_cars = []

	def perform(self):
		cars, not_cars = self.__load_dataset()
		return self._apply_output({'cars': cars, 'not_cars': not_cars})

	def cars(self):
		return self.output()['cars']

	def not_cars(self):
		return self.output()['not_cars']

	def __load_dataset(self):
		cars = self.__cars
		not_cars = self.__notcars

		if len(cars) <= 0:
			# Read in cars and not_cars
			images = glob.glob('notes/hog_images/**/*.jpeg')

			for image in images:
				image_name = os.path.basename(image)
				if 'image' in image_name or 'extra' in image_name:
					not_cars.append(image)
				else:
					cars.append(image)
			self.__cars = cars
			self.__notcars = not_cars

		return cars, not_cars


class VehicleDetectionOp(PipelineOp):
	def __init__(self, calibration_op):
		PipelineOp.__init__(self)
		self.__calibration_op = calibration_op

	def perform(self):
		return self._apply_output(None)
