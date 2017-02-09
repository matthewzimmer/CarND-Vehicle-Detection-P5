import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from skimage.feature import hog
from mpl_toolkits.mplot3d import Axes3D


class PipelineOp:
	"""
	Pipeline Operations

	Pipeline operations are the principle driving force for this project. Each implementation of `PipelineOp` is a modular,
	reusable algorithm which performs a single operation on an image. `PipelineOp` has a simple interface with
	only 3 steps to satisfy the contract:

	1. Declare a constructor with inputs necessary to perform the operation in `#perform`.

	2. Implement `#perform`

	  * This method must return `self`. This provides support to perform the op and immediately assign the call to `#output`
	    to local variables.

	  * Declared your op's output by calling `#_apply_output` once you've performed your operation.
	"""

	def __init__(self):
		self.__output = None

	def perform(self):
		raise NotImplementedError

	def output(self):
		return self.__output

	def _apply_output(self, value):
		self.__output = value
		return self


class ConvertColorSpaceOp(PipelineOp):
	def __init__(self, img, color_space, src_color_space='RGB', color_channel=-1):
		"""
		Converts an image to a different color space.

		Available color spaces: HSV, HLS, YUV, GRAY, YCrCb
		"""
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__color_space = color_space.upper()
		self.__src_color_space = src_color_space.upper()
		self.__color_channel = color_channel

	def perform(self):
		img = cv2.cvtColor(self.__img,
		                   eval('cv2.COLOR_{}2{}'.format(self.__src_color_space, self.__color_space))).astype(np.float)
		if self.__color_channel > -1:
			img = img[:, :, self.__color_channel]
		return self._apply_output(img)


class ColorThreshOp(PipelineOp):
	def __init__(self, gray_img, color_thresh=(0, 255)):
		PipelineOp.__init__(self)
		self.__img = np.copy(gray_img)
		self.__color_thresh = color_thresh

	def perform(self):
		binary = np.zeros_like(self.__img)
		binary[(self.__img > self.__color_thresh[0]) & (self.__img <= self.__color_thresh[1])] = 1
		return self._apply_output(binary)


class UndistortOp(PipelineOp):
	def __init__(self, img, camera_calibration_op):
		"""
		Takes an image and cam and performs image distortion correction
		"""
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__camera_calibration_op = camera_calibration_op

	def perform(self):
		return self._apply_output(self.__camera_calibration_op.undistort(self.__img))


class SobelThreshOp(PipelineOp):
	def __init__(self, gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
		PipelineOp.__init__(self)
		self.__img = np.copy(gray)
		self.__orient = orient
		self.__sobel_kernel = sobel_kernel  # Choose a larger odd number to smooth gradient measurements
		self.__thresh = thresh

	def perform(self):
		gray = self.__img
		sobel = cv2.Sobel(gray, cv2.CV_64F, self.__orient == 'x', self.__orient != 'x', ksize=self.__sobel_kernel)
		abs_sobel = np.absolute(sobel)
		scaled_sobel = (255 * abs_sobel / np.max(abs_sobel)).astype(np.uint8)
		binary = np.zeros_like(scaled_sobel)
		binary[(scaled_sobel >= self.__thresh[0]) & (scaled_sobel <= self.__thresh[1])] = 1
		return self._apply_output(binary)


class MagnitudeGradientThreshOp(PipelineOp):
	def __init__(self, gray_img, sobel_kernel=3, thresh=(0, 255)):
		PipelineOp.__init__(self)
		self.__img = np.copy(gray_img)
		self.__sobel_kernel = sobel_kernel  # Choose a larger odd number to smooth gradient measurements
		self.__thresh = thresh

	def perform(self):
		gray = self.__img
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.__sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.__sobel_kernel)
		gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
		gradmag = (255 * gradmag / np.max(gradmag)).astype(np.uint8)
		binary = np.zeros_like(gradmag)
		binary[(gradmag >= self.__thresh[0]) & (gradmag <= self.__thresh[1])] = 1
		return self._apply_output(binary)


class DirectionGradientThreshOp(PipelineOp):
	"""
	Calculates the gradient direction of detected lines
	"""

	def __init__(self, gray_img, sobel_kernel=3, thresh=(0, np.pi / 2)):
		PipelineOp.__init__(self)
		self.__img = np.copy(gray_img)
		self.__sobel_kernel = sobel_kernel  # Choose a larger odd number to smooth gradient measurements
		self.__thresh = thresh

	def perform(self):
		sobelx = cv2.Sobel(self.__img, cv2.CV_64F, 1, 0, ksize=self.__sobel_kernel)
		sobely = cv2.Sobel(self.__img, cv2.CV_64F, 0, 1, ksize=self.__sobel_kernel)
		with np.errstate(divide='ignore', invalid='ignore'):
			abs_grad_dir = np.absolute(np.arctan(sobely / sobelx))
			binary = np.zeros_like(abs_grad_dir)
			binary[(abs_grad_dir > self.__thresh[0]) & (abs_grad_dir < self.__thresh[1])] = 1
		return self._apply_output(binary)


class WarperOp(PipelineOp):
	def __init__(self, gray_img, src_pts, dst_pts):
		PipelineOp.__init__(self)
		self.__img = np.copy(gray_img)
		self.__src_pts = src_pts
		self.__dst_pts = dst_pts

	def perform(self):
		# Compute the perspective transform, M, given source and destination points:
		M = cv2.getPerspectiveTransform(self.__src_pts, self.__dst_pts)

		# Compute the inverse perspective transform:
		Minv = cv2.getPerspectiveTransform(self.__dst_pts, self.__src_pts)

		# Warp an image using the perspective transform, M:
		warped = cv2.warpPerspective(self.__img, M, self.__img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)

		return self._apply_output({
			'warped': warped,
			'M': M,
			'Minv': Minv,
			'src_pts': self.__src_pts,
			'dst_pts': self.__dst_pts
		})

	def __str__(self):
		s = []

		s.append(' source image shape: ')
		s.append('')
		s.append('   ' + str(self.__img.shape))
		s.append('')

		s.append(' source points: ')
		s.append('')
		s.append('   top.L: ' + str(self.__src_pts[0]))
		s.append('   bot.L: ' + str(self.__src_pts[1]))
		s.append('   bot.R: ' + str(self.__src_pts[2]))
		s.append('   top.R: ' + str(self.__src_pts[3]))
		s.append('')

		s.append(' desination points: ')
		s.append('')
		s.append('   top.L: ' + str(self.__dst_pts[0]))
		s.append('   bot.L: ' + str(self.__dst_pts[1]))
		s.append('   bot.R: ' + str(self.__dst_pts[2]))
		s.append('   top.R: ' + str(self.__dst_pts[3]))
		s.append('')
		s.append('')

		return '\n'.join(s)


class PlotImageOp(PipelineOp):
	def __init__(self, img, title='', cmap='gray', interpolation='none', aspect='auto'):
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__title = title
		self.__cmap = cmap
		self.__interpolation = interpolation
		self.__aspect = aspect

	def perform(self):
		fig1 = plt.figure(figsize=(16, 9))
		ax = fig1.add_subplot(111)
		ax.imshow(self.__img, cmap=self.__cmap, interpolation=self.__interpolation, aspect=self.__aspect)
		plt.tight_layout()
		ax.set_title(self.__title)
		plt.show()
		return self._apply_output(ax)


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


class ReshapeAndCvtColorOp(PipelineOp):
	def __init__(self, img, color_space=None, size=(32, 32)):
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__color_space = color_space
		self.__size = size

	def perform(self):
		output = self.__img

		# Convert image to new color space (if specified)
		if self.__color_space != None and self.__color_space != 'RGB':
			output = cv2.cvtColor(output, eval('cv2.COLOR_BGR2' + self.__color_space))

		# Use cv2.resize().ravel() to convert the image matrix to a vector
		if self.__size is not None:
			output = cv2.resize(output, self.__size).ravel()

		# Return the feature vector
		return self._apply_output(output)


# Define a function to compute color histogram features
class ColorHistOp(PipelineOp):
	def __init__(self, img, nbins=32, bins_range=(0, 256)):
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__nbins = nbins
		self.__bins_range = bins_range

	def perform(self):
		img = self.__img
		# Compute the histogram of the RGB channels separately
		rhist = np.histogram(img[:, :, 0], bins=self.__nbins, range=self.__bins_range)
		ghist = np.histogram(img[:, :, 1], bins=self.__nbins, range=self.__bins_range)
		bhist = np.histogram(img[:, :, 2], bins=self.__nbins, range=self.__bins_range)
		# Generating bin centers
		bin_edges = rhist[1]
		bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
		# Concatenate the histograms into a single feature vector
		# These, collectively, are now our feature vector for this particular image.
		hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
		# Return the individual histograms, bin_centers and feature vector
		return self._apply_output((rhist, ghist, bhist, bin_centers, hist_features))


class Plot3dOp(PipelineOp):
	def __init__(self, pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
		PipelineOp.__init__(self)
		self.__pixels = pixels
		self.__colors_rgb = colors_rgb
		self.__axis_labels = axis_labels
		self.__axis_limits = axis_limits

	def perform(self):
		"""Plot pixels in 3D."""

		# Create figure and 3D axes
		fig = plt.figure(figsize=(8, 8))
		ax = Axes3D(fig)

		# Set axis limits
		ax.set_xlim(*self.__axis_limits[0])
		ax.set_ylim(*self.__axis_limits[1])
		ax.set_zlim(*self.__axis_limits[2])

		# Set axis labels and sizes
		ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
		ax.set_xlabel(self.__axis_labels[0], fontsize=16, labelpad=16)
		ax.set_ylabel(self.__axis_labels[1], fontsize=16, labelpad=16)
		ax.set_zlabel(self.__axis_labels[2], fontsize=16, labelpad=16)

		# Plot pixel values with colors given in colors_rgb
		ax.scatter(
			self.__pixels[:, :, 0].ravel(),
			self.__pixels[:, :, 1].ravel(),
			self.__pixels[:, :, 2].ravel(),
			c=self.__colors_rgb.reshape((-1, 3)), edgecolors='none')

		# return Axes3D object for further manipulation
		return self._apply_output(ax)


class HOGExtractorOp(PipelineOp):
	def __init__(self, img, orient, pix_per_cell, cell_per_block, visualize=False, feature_vec=True, transform_sqrt=True):
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__orient = orient
		self.__pix_per_cell = pix_per_cell
		self.__cell_per_block = cell_per_block
		self.__visualize = visualize
		self.__feature_vec = feature_vec
		self.__transform_sqrt = transform_sqrt

	def perform(self):
		features = None
		hog_image = None
		if self.__visualize:
			# Use skimage.hog() to get both features and a visualization
			features, hog_image = hog(
				self.__img,
				orientations=self.__orient,
				pixels_per_cell=(self.__pix_per_cell, self.__pix_per_cell),
				cells_per_block=(self.__cell_per_block, self.__cell_per_block),
				visualise=self.__visualize,
				feature_vector=self.__feature_vec,
				transform_sqrt=self.__transform_sqrt
			)
		else:
			# Use skimage.hog() to get features only
			features = hog(
				self.__img,
				orientations=self.__orient,
				pixels_per_cell=(self.__pix_per_cell, self.__pix_per_cell),
				cells_per_block=(self.__cell_per_block, self.__cell_per_block),
				visualise=self.__visualize,
				feature_vector=self.__feature_vec,
				transform_sqrt=self.__transform_sqrt
			)
		return self._apply_output((features, hog_image))
