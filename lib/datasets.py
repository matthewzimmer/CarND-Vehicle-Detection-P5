import glob
import os

from lib.pipeline_ops import PipelineOp


class CarsNotCarsDatasetOp(PipelineOp):
	def __init__(self, dataset_size='small'):
		PipelineOp.__init__(self)
		self.__dataset_size = dataset_size
		self.__cars = []
		self.__notcars = []

	def perform(self):
		if self.__dataset_size == 'small':
			cars, notcars = self.__load_small_dataset()
		else:
			cars, notcars = self.__load_big_dataset()
		return self._apply_output({'cars': cars, 'notcars': notcars})

	def cars(self):
		return self.output()['cars']

	def notcars(self):
		return self.output()['notcars']

	def __load_big_dataset(self):
		cars = self.__cars
		notcars = self.__notcars
		if len(cars) <= 0:
			# Read in cars and notcars
			car_images = glob.glob('data/vehicles/**/*.png')
			for image in car_images:
				cars.append(image)
			notcar_images = glob.glob('data/non-vehicles/**/*.png')
			for image in notcar_images:
				notcars.append(image)
			self.__cars = cars
			self.__notcars = notcars
		return cars, notcars

	def __load_small_dataset(self):
		cars = self.__cars
		notcars = self.__notcars

		if len(cars) <= 0:
			# Read in cars and notcars
			images = glob.glob('notes/hog_images/**/*.jpeg')

			for image in images:
				image_name = os.path.basename(image)
				if 'image' in image_name or 'extra' in image_name:
					notcars.append(image)
				else:
					cars.append(image)
			self.__cars = cars
			self.__notcars = notcars

		return cars, notcars
