import numpy as np
import cv2
import glob
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from lib.lane_detection_ops import CameraCalibrationOp, LaneDetectionOp

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

from lib.pipeline_ops import PlotImageOp



class PipelineRunner:
	def __init__(self, lane_assist_op, vehicle_detection_op):
		self.current_frame = 0
		self.__lane_assist_op = lane_assist_op

		# TODO Implement VehicleDetectionOp
		# self.vehicle_detection_op = VehicleDetectionOp(
		# 	calibration_op
		# )

	def process_video(self, src_video_path, dst_video_path, audio=False):
		self.current_frame = 0
		VideoFileClip(src_video_path).fl_image(self.process_image).write_videofile(dst_video_path, audio=audio)

	def process_image(self, image):
		self.current_frame += 1
		return self.__lane_assist_op.process_image(
			image,
			self.current_frame
		).output()


if __name__ == '__main__':
	# I'm calibrating my camera outside of the PipelineRunner because if I decided to record my own video using my
	# camera, I'd need to calibrate it using checkerboard images taken with my camera. This just saves me from having
	# to refactor the code later (because it's easy to do it right the first time).
	calibration_images = glob.glob('camera_cal/calibration*.jpg')
	calibration_op = CameraCalibrationOp(
		calibration_images=calibration_images,
		x_inside_corners=9,
		y_inside_corners=6
	).perform()

	lane_assist_op = LaneDetectionOp(
		calibration_op,
		margin=100,
		kernel_size=15,
		sobelx_thresh=(20, 100),
		sobely_thresh=(20, 100),
		mag_grad_thresh=(20, 250),
		dir_grad_thresh=(0.3, 1.3),
		color_space='HSV',
		color_channel=2,
		processed_images_save_dir=os.path.basename(img_path).split('.')[0]
	)
	# vehicle detection manager
	# vehicle_detection_op = VehicleDetectionOp(
	# 	calibration_op
	# )

	# See how well my pipeline performs against all .jpg images inside test_images directory
	if True:
		images = glob.glob('test_images/*.jpg')
		for img_path in images:
			result = PipelineRunner(
				calibration_op,
				lane_assist_op,
			).process_image(mpimg.imread(img_path))
			# PlotImageOp(result, title="{} - FINAL".format(img_path)).perform()

	# Run pipeline against the main project_video.mp4
	if False:
		PipelineRunner(
			calibration_op,
			color_space='HSV',
			color_channel=2,
			processed_images_save_dir='project_video'
		).process_video('project_video.mp4', 'project_video_final.mp4')

	# Run pipeline against the challenge_video.mp4
	if False:
		PipelineRunner(
			calibration_op,
			color_space='HSV',
			color_channel=2,
			processed_images_save_dir='challenge_video'
		).process_video('challenge_video.mp4', 'challenge_video_final.mp4')

	# Run pipeline against the test_video.mp4
	if False:
		PipelineRunner(
			calibration_op,
			color_space='HSV',
			color_channel=2,
			processed_images_save_dir='test_video'
		).process_video('test_video.mp4', 'test_video_final.mp4')