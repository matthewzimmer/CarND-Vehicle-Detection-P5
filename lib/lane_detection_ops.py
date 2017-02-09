import numpy as np
import cv2
import os
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from lib.pipeline_ops import PipelineOp, UndistortOp, ConvertColorSpaceOp, SobelThreshOp, \
	MagnitudeGradientThreshOp, DirectionGradientThreshOp, WarperOp


class CameraCalibrationOp(PipelineOp):
	MODE_INITIALIZED = 0x0
	MODE_CALIBRATING = 0x1
	MODE_CALIBRATED = 0x2

	# Bitmask of completed camera calibration stages
	COMPLETED_CALIBRATION_STAGES = 0x0

	STAGE_OBTAINED_CALIBRATION_IMAGES = 0x1 << 0
	STAGE_COMPUTED_OBJ_AND_IMG_POINTS = 0x1 << 1
	STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS = 0x1 << 2
	STAGE_UNDISTORED_CALIBRATION_IMAGES = 0x1 << 3
	STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS = 0x1 << 4

	def __init__(self, calibration_images, x_inside_corners=9, y_inside_corners=6,
	             calibration_results_pickle_file="camera_cal/camera_mtx_and_dist_pickle.p"):
		PipelineOp.__init__(self)

		self.__mode = self.MODE_INITIALIZED

		# Images taken by a camera for which this class calibrates by
		# calculating the object and image points used to undistort
		# any image take by the same camera.
		self.__calibration_images = calibration_images

		self.__x_inside_corners = x_inside_corners
		self.__y_inside_corners = y_inside_corners

		# Arrays to store object points and image points from all the images.
		self.__objpoints = []  # 3d points in real world space
		self.__imgpoints = []  # 2d points in image plane

		# Computed using cv2.calibrateCamera() in __compute_camera_matrix_and_distortion_coefficients
		self.__camera_matrix = None
		self.__distortion_coefficients = None

		# The location of the pickle file where our camera calibration matrix and
		# distortion coefficients are persisted to
		self.__calibration_results_pickle_file = calibration_results_pickle_file

		self.__apply_stage(self.STAGE_OBTAINED_CALIBRATION_IMAGES)

	def perform(self):
		self.__mode = self.MODE_CALIBRATING
		self.__compute_obj_and_img_points()
		calibrations = self.__load_calibrations()
		if calibrations is False:
			self.__compute_camera_matrix_and_distortion_coefficients(self.__calibration_images[0])
			self.__save_calibration_mtx_and_dist()
		# self.__undistort_chessboard_images()
		self.__mode = self.MODE_CALIBRATED
		return self._apply_output({
			'matrix': self.__camera_matrix,
			'dist_coefficients': self.__distortion_coefficients,
			'objpoints': self.__objpoints,
			'imgpoints': self.__imgpoints
		})

	def undistort(self, img):
		"""
		A function that takes an image and performs the camera calibration,
		image distortion correction and returns the undistorted image
		"""
		img = np.copy(img)
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
			self.__objpoints,
			self.__imgpoints,
			img.shape[0:2][::-1],
			None,
			None
		)
		return cv2.undistort(img, mtx, dist, None, mtx)

	# PRIVATE

	def __detect_corners(self, img, nx, ny):
		"""
		This function converts an RGB chessboard image to grayscale and finds the
		chessboard corners using cv2.findChessboardCorners.
		"""
		img = np.copy(img)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		return cv2.findChessboardCorners(gray, (nx, ny), None)

	def __compute_obj_and_img_points(self):
		"""
		A function which iterates over all self.calibration_images and detects all
		chessboard corners for each.

		For each image corners are detected, a copy of that image with the corners
		drawn on are saved to camera_cal/corners_found
		"""
		if not self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS):
			nx, ny = self.__x_inside_corners, self.__y_inside_corners

			# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
			objp = np.zeros((nx * ny, 3), np.float32)
			objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

			# Step through the list and search for chessboard corners
			for fname in self.__calibration_images:
				img = mpimg.imread(fname)
				ret, corners = self.__detect_corners(img, nx, ny)

				# If found, add object points, image points
				if ret:
					self.__objpoints.append(objp)
					self.__imgpoints.append(corners)

					# print("{} corners detected".format(os.path.basename(fname)))
					calibrated_name = 'camera_cal/corners_found/{}'.format(str(os.path.basename(fname)))
					cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
					cv2.imwrite(calibrated_name, img)

			self.__apply_stage(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS)

	def __undistort_chessboard_images(self):
		if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) and not self.__is_stage_complete(
				self.STAGE_UNDISTORED_CALIBRATION_IMAGES):
			# Step through the list and search for chessboard corners
			for fname in self.__calibration_images:
				img = mpimg.imread(fname)
				undistorted = self.undistort(img)

				# print("{} undistorted".format(os.path.basename(fname)))
				undist_file = 'camera_cal/undistorted/{}'.format(os.path.basename(fname))
				cv2.imwrite(undist_file, undistorted)

			self.__apply_stage(self.STAGE_UNDISTORED_CALIBRATION_IMAGES)

	def __load_calibrations(self):
		if os.path.isfile(self.__calibration_results_pickle_file):
			with open(self.__calibration_results_pickle_file, 'rb') as f:
				pickle_data = pickle.load(f)
				self.__camera_matrix = pickle_data['mtx']
				self.__distortion_coefficients = pickle_data['dist']
				if self.__camera_matrix is not None:
					self.__apply_stage(self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS)
					self.__apply_stage(self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS)
					return True
		return False

	def __compute_camera_matrix_and_distortion_coefficients(self, distorted_image_path):
		if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) and not self.__is_stage_complete(
				self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS):
			fname = distorted_image_path
			img = mpimg.imread(fname)
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.__objpoints, self.__imgpoints, img.shape[0:2][::-1],
			                                                   None, None)

			self.__camera_matrix = mtx
			self.__distortion_coefficients = dist

			self.__apply_stage(self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS)

	def __save_calibration_mtx_and_dist(self):
		"""
		Saves a pickled representation of the camera calibration matrix and
		distortion coefficient results for the provided image for later use
		"""
		if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) and not self.__is_stage_complete(
				self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS):
			dist_pickle = {}
			dist_pickle["mtx"] = self.__camera_matrix
			dist_pickle["dist"] = self.__distortion_coefficients
			pickle.dump(dist_pickle, open(self.__calibration_results_pickle_file, "wb"))

			# print('camera matrix and distortion coefficients pickled to "{}" for later use'.format(self.__calibration_results_pickle_file))

			self.__apply_stage(self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS)

	def __is_stage_complete(self, flag):
		return self.COMPLETED_CALIBRATION_STAGES & flag == flag

	def __apply_stage(self, flag):
		"""Marks a stage as complete"""
		self.COMPLETED_CALIBRATION_STAGES = self.COMPLETED_CALIBRATION_STAGES | flag

	def __str__(self):
		s = []

		s.append('')
		s.append('')
		s.append('-------------------------------------------------------------')
		s.append('')
		s.append('[ CALIBRATION MODES ]')
		s.append('')
		s.append('   Initialized? {}'.format('YES' if self.__mode == self.MODE_INITIALIZED else 'NO'))
		s.append('   Calibrating? {}'.format('YES' if self.__mode == self.MODE_CALIBRATING else 'NO'))
		s.append('   Calibration complete? {}'.format('YES' if self.__mode == self.MODE_CALIBRATED else 'NO'))
		s.append('')
		s.append('')

		s.append('[ CALIBRATION STAGES - {} ]'.format(self.COMPLETED_CALIBRATION_STAGES))
		s.append('')
		s.append('   Obtained calibration images? {}'.format(
			'YES' if self.__is_stage_complete(self.STAGE_OBTAINED_CALIBRATION_IMAGES) else 'NO'))
		s.append('   Computed object/image points? {}'.format(
			'YES' if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) else 'NO'))
		s.append('   Calculated camera matrix and distortion coefficients? {}'.format(
			'YES' if self.__is_stage_complete(self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS) else 'NO'))
		s.append('   Undistored calibration images? {}'.format(
			'YES' if self.__is_stage_complete(self.STAGE_UNDISTORED_CALIBRATION_IMAGES) else 'NO'))
		s.append('   Persisted camera matrix and distortion coefficients? {}'.format(
			'YES' if self.__is_stage_complete(self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS) else 'NO'))
		s.append('')

		s.append('[ PARAMS ]')
		s.append('')
		s.append('Number calibration images: {}'.format(len(self.__calibration_images)))
		s.append('X inside corners = {}'.format(self.__x_inside_corners))
		s.append('Y inside corners = {}'.format(self.__y_inside_corners))
		s.append('')
		# s.append('output = {}'.format(str(self.output())))

		s.append('')
		s.append('')

		return '\n'.join(s)


class DrawPolyLinesOp(PipelineOp):
	def __init__(self, img, pts, color=(0, 140, 255), thickness=5):
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__pts = pts
		self.__color = color
		self.__thickness = thickness

	def perform(self):
		return self._apply_output(cv2.polylines(self.__img, [np.array([self.__pts], np.int32)], True, self.__color,
		                                        thickness=self.__thickness))


class PolyfitLineOp(PipelineOp):
	def __init__(self, binary_warped):
		PipelineOp.__init__(self)
		self.__binary_warped = binary_warped

	def perform(self):
		binary_warped = self.__binary_warped

		# Takes a histogram of the bottom half of a warped binary image
		histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)

		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0] / 2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Choose the number of sliding windows
		nwindows = 9
		# Set height of windows
		window_height = np.int(binary_warped.shape[0] / nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		margin = 100
		# Set minimum number of pixels found to recenter window
		minpix = 50
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window + 1) * window_height
			win_y_high = binary_warped.shape[0] - window * window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			# Draw the windows on the visualization image
			cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
			cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
				nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
				nonzerox < win_xright_high)).nonzero()[0]
			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		return self._apply_output({
			'left_fit': left_fit,
			'right_fit': right_fit,
			'out_img': out_img,
			'left_lane_inds': left_lane_inds,
			'right_lane_inds': right_lane_inds,
			'leftx': leftx,
			'lefty': lefty,
			'rightx': rightx,
			'righty': righty,
			'nonzerox': nonzerox,
			'nonzeroy': nonzeroy
		})


# A class to receive the characteristics of each line detection
#
#   **** >>>> UNIMPLEMENTED...NOT USED BY MY PIPELINE FOR ANYTHING WORTH NOTING
#
class LineOp(PipelineOp):
	# This constant ultimately contributes to deriving a given
	# period when computing SMA and EMA for line noise smoothing
	FPS = 30

	def __init__(self, ema_period_alpha=0.65):
		PipelineOp.__init__(self)
		self.__ema_fps_period = ema_period_alpha * self.FPS
		self.__all_measurements = np.array([])

		# was the line detected in the last iteration?
		self.detected = False
		# x values of the last n fits of the line
		self.recent_xfitted = []
		# average x values of the fitted line over the last n iterations
		self.bestx = None
		# polynomial coefficients averaged over the last n iterations
		self.best_fit = None
		# polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]
		# radius of curvature of the line in some units
		self.radius_of_curvature = None
		# distance in meters of vehicle center from the line
		self.line_base_pos = None
		# difference in fit coefficients between last and new fits
		self.diffs = np.array([0, 0, 0], dtype='float')
		# x values for detected line pixels
		self.allx = np.array([])
		# y values for detected line pixels
		self.ally = np.array([])

	def perform(self):
		# self.__compute_ema(x, self.allx)
		return self

	def add_measurement(self, x, y, radius_of_curvature, line_base_pos):
		self.__all_measurements = np.append(self.__all_measurements, (x, y, radius_of_curvature, line_base_pos))
		self.allx = np.append(self.allx, x)
		self.ally = np.append(self.ally, y)
		self.radius_of_curvature = radius_of_curvature
		self.line_base_pos = line_base_pos
		return self

	def __compute_ema(self, measurement, all_measurements, curr_ema):
		sma = sum(all_measurements) / (len(all_measurements))

		if len(all_measurements) < self.__ema_fps_period:
			# let's just use SMA until
			# our EMA buffer is filled
			return sma

		multiplier = 2 / float(len(all_measurements) + 1)
		ema = (measurement - curr_ema) * multiplier + curr_ema

		# print("sma: %s, multiplier: %s" % (sma, multiplier))
		return ema


class LaneDetectionOp(PipelineOp):
	def __init__(
			self,
			calibration_op,
			margin=100,
			kernel_size=3,
			color_thresh=(205, 255),
			sobelx_thresh=(20, 100),
			sobely_thresh=(20, 100),
			mag_grad_thresh=(20, 250),
			dir_grad_thresh=(0., np.pi / 2),
			color_space='HSV',
			color_channel=2,
			processed_images_save_dir=None
	):
		PipelineOp.__init__(self)
		self.__margin = margin
		self.__calibration_op = calibration_op
		self.__kernel_size = kernel_size
		self.__color_thresh = color_thresh
		self.__sobelx_thresh = sobelx_thresh
		self.__sobely_thresh = sobely_thresh
		self.__mag_grad_thresh = mag_grad_thresh
		self.__dir_grad_thresh = dir_grad_thresh
		self.__left_line = LineOp()
		self.__right_line = LineOp()
		self.__save_counter = 0
		self.__polyfit_op = None
		self.__color_space = color_space
		self.__color_channel = color_channel
		self.__processed_images_save_dir = processed_images_save_dir

	def __save_image(self, img, name):
		self.__save_counter += 1
		if self.__processed_images_save_dir != None:
			basedir = 'processed_images/{}'.format(self.__processed_images_save_dir)
			if not os.path.exists(basedir):
				os.makedirs(basedir)
			cv2.imwrite(basedir+'/{}_{}_{}.jpg'.format(self.__processed_images_save_dir, self.__name,
			                                                     self.__save_counter, name), img)

	def process_image(self, img, name):
		self.__save_counter = 0
		self.__img = np.copy(img)
		self.__name = name
		return self.perform()

	def perform(self):
		img = self.__img
		kernel_size = self.__kernel_size
		color_thresh = self.__color_thresh
		sobelx_thresh = self.__sobelx_thresh
		sobely_thresh = self.__sobely_thresh
		mag_grad_thresh = self.__mag_grad_thresh
		dir_grad_thresh = self.__dir_grad_thresh

		self.__save_image(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 'raw_input')

		# undistort the raw image
		undistorted = UndistortOp(img, self.__calibration_op).perform().output()
		self.__save_image(cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR), 'undistorted')

		# Convert undistored image to HSV and use the 'V' channel as our gray image.
		color_cvt = ConvertColorSpaceOp(
			undistorted,
			color_space=self.__color_space,
			color_channel=self.__color_channel
		).perform().output()
		self.__save_image(color_cvt, '{} channel {}'.format(self.__color_space, self.__color_channel))

		# NO COLOR THRESHOLDING IS DONE IN THIS PIPELINE
		# color_cvt = ColorThreshOp(color_cvt, color_thresh=color_thresh).perform().output()

		# Compute sobel X binary image
		gradx = SobelThreshOp(color_cvt, orient='x', sobel_kernel=kernel_size,
		                      thresh=sobelx_thresh).perform().output()
		self.__save_image(gradx * 255, 'gradx')

		# Compute sobel Y binary image
		grady = SobelThreshOp(
			color_cvt,
			orient='y',
			sobel_kernel=kernel_size,
			thresh=sobely_thresh
		).perform().output()
		self.__save_image(grady * 255, 'grady')

		# Compute Magnitude Gradient binary image
		mag_binary = MagnitudeGradientThreshOp(
			color_cvt,
			sobel_kernel=kernel_size,
			thresh=mag_grad_thresh
		).perform().output()
		self.__save_image(mag_binary * 255, 'mag_binary')

		# Compute Direction Gradient binary image
		dir_binary = DirectionGradientThreshOp(color_cvt, sobel_kernel=kernel_size,
		                                       thresh=dir_grad_thresh).perform().output()
		self.__save_image(dir_binary * 255, 'dir_binary')

		# Perform bitwise AND and OR to create a final binary image where we generate a binary image of
		# all white pixels in (SobelX AND SobelY) and combine it via binary OR with a binary image of all white pixels
		# in (Magnitude AND Direction) gradients.
		combined = np.zeros_like(color_cvt)
		combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
		self.__save_image(combined * 255, 'final_binary_thresh')

		# Now we're going to warp our combined threshholded binary image
		img_size = (img.shape[1], img.shape[0])

		src_pts = np.float32(
			[[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
			 [((img_size[0] / 6) - 55), img_size[1]],
			 [(img_size[0] * 5 / 6) + 60, img_size[1]],
			 [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])

		dst_pts = np.float32(
			[[(img_size[0] / 6), 0],
			 [(img_size[0] / 6), img_size[1]],
			 [(img_size[0] * 5 / 6), img_size[1]],
			 [(img_size[0] * 5 / 6), 0]])

		warper_op = WarperOp(combined, src_pts, dst_pts).perform().output()
		binary_warped = warper_op['warped']
		self.__save_image(binary_warped * 255, 'binary_warped')

		if True or self.__polyfit_op is None:
			self.__polyfit_op = PolyfitLineOp(binary_warped).perform().output()

		left_fit = self.__polyfit_op['left_fit']
		right_fit = self.__polyfit_op['right_fit']
		nonzeroy = self.__polyfit_op['nonzeroy']
		nonzerox = self.__polyfit_op['nonzerox']
		left_lane_inds = self.__polyfit_op['left_lane_inds']
		right_lane_inds = self.__polyfit_op['right_lane_inds']

		# Generate x and y values for plotting
		fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
		fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
		fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

		out_img = self.__polyfit_op['out_img']
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# Assume you now have a new warped binary image
		# from the next frame of video (also called "binary_warped")
		# It's now much easier to find line pixels!
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		margin = self.__margin
		left_lane_inds = (
			(nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
				nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

		right_lane_inds = (
			(nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
				nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

		# Again, extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		# Generate x and y values for plotting
		fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
		fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
		fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

		# Create an image to draw on and an image to show the selection window
		out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
		window_img = np.zeros_like(out_img)
		# Color in left and right line pixels
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([fit_leftx - margin, fity]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_leftx + margin, fity])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([fit_rightx - margin, fity]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_rightx + margin, fity])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Define y-value where we want radius of curvature
		# I'll choose the maximum y-value, corresponding to the bottom of the image
		y_eval = out_img.shape[0]

		left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
		right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(
			2 * right_fit[0])

		# print('left:', left_curverad, '| right:', right_curverad)

		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 30 / out_img.shape[0]  # meters per pixel in y dimension
		xm_per_pix = 3.7 / (out_img.shape[0] - 20)  # meteres per pixel in x dimension

		# Find center of car relitive to center of lane
		# determine center of the lane.
		left_line_baseX = left_fit[2]  # Lesson 12 - distance from the bottom(y=720) left(x=0) of the image to the left lane line in pixels.
		left_line_baseMeters = left_line_baseX * xm_per_pix
		right_line_baseX = right_fit[2]  # same for the right lane line.
		right_line_baseMeters = right_line_baseX * xm_per_pix
		lane_center_pixels = left_line_baseX + (0.5 * (right_line_baseX - left_line_baseX))  # in pixels
		# convert to meters
		lane_center_meters = lane_center_pixels * xm_per_pix  # pixels*meter/pixel = meter
		car_center_pixels = (out_img.shape[1] / 2) + 43
		# Convert to meters.
		# meters from left of image
		car_center_meters = car_center_pixels * xm_per_pix
		# Car center with respect to lane center in meters
		car_relative_position = lane_center_meters - car_center_meters
		# print(car_relative_position, 'car_relative_position meters')

		left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
		right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

		left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
		right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
		# Now our radius of curvature is in meters
		# print('left curve rad: {}m    |     right curve rad: {}m'.format(left_curverad, right_curverad))


		# Create an image to draw the lines on
		warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
		# pts_right = np.array([np.transpose(np.vstack([fit_rightx, fity]))])
		pts = np.hstack((pts_left, pts_right))

		self.__left_line.add_measurement(fit_leftx, fity, left_curverad, 0).perform()
		self.__right_line.add_measurement(fit_rightx, fity, right_curverad, 0).perform()

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0, 195, 255))

		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = cv2.warpPerspective(
			color_warp,
			warper_op['Minv'],
			(binary_warped.shape[1], binary_warped.shape[0])
		)

		# Combine the result with the original image
		result = cv2.addWeighted(undistorted, 1, newwarp, 0.5, 0)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(result, "<--[   CURVE RADIUS   ]-->", (535, 710), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(result, '{0:.2f}m'.format(left_curverad), (380, 710), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(result, '{0:.2f}m'.format(right_curverad), (850, 710), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

		cv2.putText(result, "Car Relative to Center", (550, 525), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(
			result,
			str('{0:.2f}m'.format(car_relative_position)),
			(600, 500),
			font,
			0.95,
			(255, 255, 255),
			2,
			cv2.LINE_AA
		)

		self.__save_image(cv2.cvtColor(result, cv2.COLOR_RGB2BGR), 'FINAL_LANE')

		return self._apply_output(result)
