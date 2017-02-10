import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from lib.pipeline_ops import PipelineOp


class VehicleDetectionManager(PipelineOp):
	def __init__(self):
		PipelineOp.__init__(self)

	def perform(self):
		return self._apply_output(None)
