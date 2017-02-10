import glob
import os

from lib.pipeline_ops import PipelineOp


class CarsNotCarsDatasetOp(PipelineOp):
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
		not_cars = self.__not_cars

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
			self.__not_cars = not_cars

		return cars, not_cars
