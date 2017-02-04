import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pipeline_ops import PipelineOp

class DrawBoxesOp(PipelineOp):
    def __init__(self, img, bboxes, color=(0, 140, 255), thickness=5):
        PipelineOp.__init__(self)
        self.__img = img
        self.__bboxes = bboxes
        self.__color = color
        self.__thickness = thickness
    
    def perform(self):
	    # Make a copy of the image
	    output = np.copy(self.__img)
	    # Iterate through the bounding boxes
	    for pt1, pt2 in self.__bboxes:
	        # Draw a rectangle given bbox coordinates
	        cv2.rectangle(output, pt1, pt2, self.__color, thickness=self.__thickness)
        return self._apply_output(output)