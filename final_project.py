import numpy as np
import cv2
import glob
import os
import pickle

import matplotlib

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib.pipeline_ops import *
from lib.vehicle_detection_ops import *
from lib.lane_detection_ops import *
from lib.datasets import *

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

calibration_op = None


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True, transform_sqrt=True):
	# Call with two outputs if vis==True
	#     if vis == True:
	#         features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
	#                                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
	#                                   visualise=vis, feature_vector=feature_vec)
	#         return features, hog_image
	#     # Otherwise call with one output
	#     else:
	#         features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
	#                        cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
	#                        visualise=vis, feature_vector=feature_vec)
	#     return features
	return HOGExtractorOp(
		img,
		orient=orient,
		pix_per_cell=pix_per_cell,
		cell_per_block=cell_per_block,
		visualize=vis,
		feature_vec=feature_vec,
		transform_sqrt=transform_sqrt
	).output()[0]


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		file_features = []
		# Read in each one by one
		image = mpimg.imread(file)
		# apply color conversion if other than 'RGB'
		if color_space != 'RGB':
			if color_space == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif color_space == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif color_space == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif color_space == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif color_space == 'YCrCb':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
		else:
			feature_image = np.copy(image)

		if spatial_feat == True:
			spatial_features = bin_spatial(feature_image, size=spatial_size)
			file_features.append(spatial_features)
		if hist_feat == True:
			# Apply color_hist()
			hist_features = color_hist(feature_image, nbins=hist_bins)
			file_features.append(hist_features)
		if hog_feat == True:
			# Call get_hog_features() with vis=False, feature_vec=True
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:, :, channel],
					                                     orient, pix_per_cell, cell_per_block,
					                                     vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)
			else:
				hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
				                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			# Append the new feature vector to the features list
			file_features.append(hog_features)
		features.append(np.concatenate(file_features))
	# Return list of feature vectors
	return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel()
	# Return the feature vector
	return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
	channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
	channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	# Compute the span of the region to be searched
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
	# Compute the number of windows in x/y
	nx_windows = np.int(xspan / nx_pix_per_step) - 1
	ny_windows = np.int(yspan / ny_pix_per_step) - 1
	# Initialize a list to append window positions to
	window_list = []
	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs * nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys * ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	# Return the list of windows
	return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	# Make a copy of the image
	imcopy = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
		# Draw a rectangle given bbox coordinates
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	# Return the image copy with boxes drawn
	return imcopy


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
	# 1) Define an empty list to receive features
	img_features = []
	# 2) Apply color conversion if other than 'RGB'
	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else:
		feature_image = np.copy(img)
	# 3) Compute spatial features if flag is set
	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		# 4) Append features to list
		img_features.append(spatial_features)
	# 5) Compute histogram features if flag is set
	if hist_feat == True:
		hist_features = color_hist(feature_image, nbins=hist_bins)
		# 6) Append features to list
		img_features.append(hist_features)
	# 7) Compute HOG features if flag is set
	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(get_hog_features(feature_image[:, :, channel],
				                                     orient, pix_per_cell, cell_per_block,
				                                     vis=False, feature_vec=True))
		else:
			hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
			                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		# 8) Append features to list
		img_features.append(hog_features)

	# 9) Return concatenated array of features
	return np.concatenate(img_features)


def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1] + 1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
	# Return the image
	return img


def plot_histogram(image, nbins=32, bins_range=(0, 256), title=''):
	ch1h, ch2h, ch3h, bincen, feature_vec = ColorHistOp(image, nbins=nbins, bins_range=bins_range).perform().output()

	# Plot a figure with all three bar charts
	if ch1h is not None:
		fig = plt.figure(figsize=(12, 3))
		plt.subplot(131)
		plt.bar(bincen, ch1h[0])
		plt.xlim(0, 256)
		plt.title(title + ' Ch1 Histogram')
		plt.subplot(132)
		plt.bar(bincen, ch2h[0])
		plt.xlim(0, 256)
		plt.title(title + ' Ch2 Histogram')
		plt.subplot(133)
		plt.bar(bincen, ch3h[0])
		plt.xlim(0, 256)
		plt.title(title + ' Ch3 Histogram')
		fig.tight_layout()
	else:
		print('Your function is returning None for at least one variable...')


def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap


def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap


class Params():
	def __init__(
			self,
			colorspace='YCrBr',
			orient=9,
			pix_per_cell=4,
			cell_per_block=4,
			hog_channel='ALL',
			spatial_size=(64, 64),
			hist_bins=64,
			spatial_feat=True,
			hist_feat=True,
			hog_feat=True
	):
		self.colorspace = colorspace  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
		self.orient = orient  # typically between 6 and 12
		self.pix_per_cell = pix_per_cell  # HOG pixels per cell
		self.cell_per_block = cell_per_block  # HOG cells per block
		self.hog_channel = hog_channel  # Can be 0, 1, 2, or "ALL"
		self.spatial_size = spatial_size  # Spatial binning dimensions
		self.hist_bins = hist_bins  # Number of histogram bins
		self.spatial_feat = spatial_feat  # Spatial features on or off
		self.hist_feat = hist_feat  # Histogram features on or off
		self.hog_feat = hog_feat  # HOG features on or off


def calibrate_camera():
	global calibration_op

	if calibration_op == None:
		# base edges - doesn't work for all images in camera_cal directory (i.e., 1, 4, 5)
		calibration_images = glob.glob('camera_cal/calibration*.jpg')

		# I will now inject this calibration_op instance later on
		# into my pipeline principally used to undistort the
		# raw image.
		calibration_op = CameraCalibrationOp(
			calibration_images=calibration_images,
			x_inside_corners=9,
			y_inside_corners=6
		).perform()
	return calibration_op


def train_classifier(params, test_cars, test_notcars, C=1.):
	t = time.time()
	car_features = extract_features(test_cars, color_space=params.colorspace,
	                                spatial_size=params.spatial_size, hist_bins=params.hist_bins,
	                                orient=params.orient, pix_per_cell=params.pix_per_cell,
	                                cell_per_block=params.cell_per_block,
	                                hog_channel=params.hog_channel, spatial_feat=params.spatial_feat,
	                                hist_feat=params.hist_feat, hog_feat=params.hog_feat)

	notcar_features = extract_features(test_notcars, color_space=params.colorspace,
	                                   spatial_size=params.spatial_size, hist_bins=params.hist_bins,
	                                   orient=params.orient, pix_per_cell=params.pix_per_cell,
	                                   cell_per_block=params.cell_per_block,
	                                   hog_channel=params.hog_channel, spatial_feat=params.spatial_feat,
	                                   hist_feat=params.hist_feat, hog_feat=params.hog_feat)
	t2 = time.time()
	print(round(t2 - t, 2), 'Seconds to extract HOG features...')

	X = np.vstack((car_features, notcar_features)).astype(np.float64)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Using:', params.orient, 'orientations', params.pix_per_cell,
	      'pixels per cell and', params.cell_per_block, 'cells per block')
	print('Feature vector length:', len(X_train[0]))
	# Use a linear SVC
	# svc = LinearSVC(C=1.2) # 98.76%
	svc = LinearSVC(C=C)
	# from sklearn.ensemble import AdaBoostClassifier
	# svc = AdaBoostClassifier(learning_rate=0.1, algorithm='SAMME.R', n_estimators=50)
	# Check the training time for the SVC
	t = time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2 - t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

	return svc, X_scaler


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
	X_pred = []

	# 1) Create an empty list to receive positive detection windows
	on_windows = []
	# 2) Iterate over all windows in the list
	for window in windows:
		# 3) Extract the test window from original image
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		# 4) Extract features for that window using single_img_features()
		features = single_img_features(test_img, color_space=color_space,
		                               spatial_size=spatial_size, hist_bins=hist_bins,
		                               orient=orient, pix_per_cell=pix_per_cell,
		                               cell_per_block=cell_per_block,
		                               hog_channel=hog_channel, spatial_feat=spatial_feat,
		                               hist_feat=hist_feat, hog_feat=hog_feat)
		X_pred.append(features)
	# 5) Predict car or notcar
	predictions = clf.predict(scaler.transform(np.array(X_pred)))
	for i, prediction in enumerate(predictions):
		if prediction == 1:
			on_windows.append(windows[i])

	# 8) Return windows for positive detections
	return on_windows


class PipelineRunner:
	def __init__(self, calibration_op, svc, X_scaler, params, color_space='HSV', color_channel=2):
		self.current_frame = -1
		self.svc = svc
		self.X_scaler = X_scaler
		self.params = params
		self.heat = None

		self.__processed_images_subdir = ''
		self.lane_assist_op = LaneDetectionOp(
			calibration_op,
			margin=100,
			kernel_size=15,
			sobelx_thresh=(20, 100),
			sobely_thresh=(20, 100),
			mag_grad_thresh=(20, 250),
			dir_grad_thresh=(0.3, 1.3),
			color_space=color_space,
			color_channel=color_channel
		)

	def process_video(self, src_video_path, dst_video_path, audio=False):
		self.current_frame = -1
		self.__processed_images_subdir = os.path.basename(src_video_path).split('.')[0] + '/'
		VideoFileClip(src_video_path).fl_image(self.process_image).write_videofile(dst_video_path, audio=audio)

	def process_image(self, image):
		self.current_frame += 1
		cv2.imwrite('processed_images/{}{}_0_IN.jpg'.format(self.__processed_images_subdir, self.current_frame),
		            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

		image = self.__detect_vehicles(image)

		#         image = self.lane_assist_op.process_image(
		#             image,
		#             '{}{}'.format(self.__processed_images_subdir, self.current_frame)
		#         ).output()

		cv2.imwrite('processed_images/{}{}_0_OUT.jpg'.format(self.__processed_images_subdir, self.current_frame),
		            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

		return image

	def __detect_vehicles(self, image):
		svc = self.svc
		X_scaler = self.X_scaler
		params = self.params

		# Uncomment the following line if you extracted training
		# data from .png images (scaled 0 to 1 by mpimg) and the
		# image you are searching is a .jpg (scaled 0 to 255)
		image = image.astype(np.float32) / 255

		windows = []
		for w_size in list([180, 120]):
			# for w_size in list([300,150,125]):
			found_windows = slide_window(image, x_start_stop=[256, 1152], y_start_stop=[375, 695],
			                             xy_window=(int(w_size * (1.5)), w_size), xy_overlap=(0.88, 0.88))
			print('Found {} windows for {} window...'.format(len(found_windows), str(w_size)))
			windows += found_windows

		t = time.time()
		hot_windows = search_windows(image, windows, svc, X_scaler, color_space=params.colorspace,
		                             spatial_size=params.spatial_size, hist_bins=params.hist_bins,
		                             orient=params.orient, pix_per_cell=params.pix_per_cell,
		                             cell_per_block=params.cell_per_block,
		                             hog_channel=params.hog_channel, spatial_feat=params.spatial_feat,
		                             hist_feat=params.hist_feat, hog_feat=params.hog_feat)
		t2 = time.time()
		print(round(t2 - t, 2), 'Seconds to search and locate {} windows...'.format(len(hot_windows)))

		t = time.time()
		window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=3)
		t2 = time.time()
		#     print(round(t2-t, 2), 'Seconds to draw boxes...')

		# PlotImageOp(window_img, title="Detected Vehicles").perform()

		base_thresh = 7
		if True or self.heat == None or (self.current_frame % base_thresh) == 0:
			print('  NEW HEATMAP!!')
			self.heat = np.zeros_like(image[:, :, 0]).astype(np.float)
		heat = self.heat

		heat_thresh = (base_thresh + (
		(base_thresh * (self.current_frame % base_thresh)) * ((self.current_frame % base_thresh) / base_thresh)))
		print('current heat threshold: {}'.format(heat_thresh))
		heatmap = add_heat(heat, hot_windows)
		heatmap = apply_threshold(heatmap, 4)
		labels = label(heatmap)
		print(labels[1], 'cars found')
		# plt.imshow(labels[0], cmap='gray')
		# plt.show()
		#         PlotImageOp(labels[0], cmap='gray').perform()
		# print(labels)

		final_map = np.clip(heatmap - 2, 0, 255)
		# PlotImageOp(final_map, cmap='hot').perform()

		draw_img = draw_labeled_bboxes(np.copy(image * 255), labels)
		# Display the image
		# PlotImageOp(draw_img, cmap=None).perform()

		return draw_img


if __name__ == '__main__':
	calibrate_camera()
	ds = CarsNotCarsDatasetOp(dataset_size='big').perform()
	cars = ds.cars()
	notcars = ds.notcars()

	# Reduce the sample size because
	# The quiz evaluator times out after 13s of CPU time
	sample_size = 1000
	c_random_idxs = np.random.randint(0, len(cars), sample_size)
	nc_random_idxs = np.random.randint(0, len(notcars), sample_size)
	test_cars = cars  # np.array(cars)[c_random_idxs] # cars[0:c_sample_size]
	test_notcars = notcars  # np.array(notcars)[nc_random_idxs] # notcars[0:nc_sample_size]

	print('    # Cars: ' + str(len(test_cars)))
	print('# Not cars: ' + str(len(test_notcars)))

	params = Params(
		colorspace='YCrCb',
		orient=7,
		pix_per_cell=8,
		cell_per_block=2,
		hog_channel='ALL',
		spatial_size=(32, 32),
		hist_bins=32,
		spatial_feat=True,
		hist_feat=True,
		hog_feat=True
	)  # 99.21%

	svc, X_scaler = train_classifier(params, test_cars, test_notcars, C=0.001)

	# See how well my pipeline performs against all .jpg images inside test_images directory
	if True:
		images = []
		images += ['processed_images/project_video/340_0_OUT.jpg']
		# images += ['processed_images/project_video/738_0_OUT.jpg']
		# images += ['notes/bbox-example-image.jpg']
		# images += glob.glob('test_images/*.jpg')
		# images += glob.glob('test_images/test3.jpg')
		for img_path in images:
			result = PipelineRunner(calibration_op, svc, X_scaler, params, color_space='HLS',
			                        color_channel=2).process_image(mpimg.imread(img_path))
			#PlotImageOp(result * 255, title="{} - FINAL".format(img_path), cmap=None).perform()

		# Run pipeline against the main project_video.mp4
	if False:
		PipelineRunner(calibration_op, svc, X_scaler, params, color_space='HSV', color_channel=2).process_video(
			'test_video.mp4', 'test_video_final.mp4')

	# Run pipeline against the main project_video.mp4
	if False:
		PipelineRunner(calibration_op, svc, X_scaler, params, color_space='HSV', color_channel=2).process_video(
			'project_video.mp4', 'project_video_final.mp4')

	# Run pipeline against the challenge_video.mp4
	if False:
		PipelineRunner(calibration_op, svc, X_scaler, params, color_space='HSV', color_channel=2).process_video(
			'challenge_video.mp4', 'challenge_video_final.mp4')
