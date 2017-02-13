
## Project 5 - Vehicle Detection and Tracking

##### By Matthew Zimmer - Future Self-Driving Car Engineer

[GitHub](https://github.com/matthewzimmer) | [LinkedIn](https://www.linkedin.com/in/matthewazimmer)

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

> **CRITICAL NOTE** I treated this notebook as a comprehensive tutorial walking you from start to finish, top down, starting from calibrating the camera all the way to the very last step of drawing the detected lane region into the road (carryover from Project 4) and drawing boxes around detected vehicles. The code cell inside the **Discussion** section of this notebook offers you the ability to control the hyper parameters used by my pipeline and either test them on a single image (the default setting), the test video, and/or the project video.

> **CRITICAL NOTE** See [classroom-notes.ipynb](./classroom-notes.ipynb) which contains all of my notes taken and tested in real-time against the various sample images as I watched each lesson. This essentially gave me the starting points to work off of for this final project.

---

#### Python imports


```python
import numpy as np
import cv2
import glob
import os
import pickle

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

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

from IPython.core.display import Image, display
from IPython.display import YouTubeVideo
def render_youtube_video(video_id, width=880, height=495):
    return YouTubeVideo(video_id, width=width, height=height)
```

---

#### Pipeline Operations

Pipeline operations are the principle driving force for this project. Each implementation of `PipelineOp` is a modular, reusable algorithm which, in its most basic form, performs a single operation on an image.

**[PipelineOp](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/107239a1530cc3b1f56016483941b28e5b70c26a/lib/pipeline_ops.py#L10-L39)** has a simple interface with only 2 requirements to satisfy the contract:

1. Declare a constructor with inputs necessary to perform the operation. To truly adhere to the nature of encapsulation and immutability, I initialize private variables so as not to expose them publicly.

2. Implement your **[#perform](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/107239a1530cc3b1f56016483941b28e5b70c26a/lib/pipeline_ops.py#L31-L32)** method

    * Your implementation must `return self`. This provides support to perform the op and immediately assign a chained call to `#output` to a local variable. Example:
    
    ```python
    out = PipelineOp().output()
    ```

    * Declare your op's final output by calling **[#_apply_output](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/107239a1530cc3b1f56016483941b28e5b70c26a/lib/pipeline_ops.py#L37-L39)** once you've performed your operation. Note the value and data type of output is arbitraily defined by your operation. Documenting this information is encouraged.
    
This architecture provides flexibility to implementing more complicated algorithms that have many moving parts while still adhering to the contract by producing a single arbitrary output object (e.g., Dictionary, Image, Array). I demonstrate a healthy mixture of both simple and complex **[PipelineOp](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/107239a1530cc3b1f56016483941b28e5b70c26a/lib/pipeline_ops.py#L10-L39)** implementations in this project. For example, **[CameraCalibrationOp](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/107239a1530cc3b1f56016483941b28e5b70c26a/lib/lane_detection_ops.py#L12-L230)** and **[LaneAssistOp](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/107239a1530cc3b1f56016483941b28e5b70c26a/lib/lane_detection_ops.py#L402-L677)** are great examples of a complex algorithm whereas **[ColorThreshOp](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/107239a1530cc3b1f56016483941b28e5b70c26a/lib/pipeline_ops.py#L63-L72)** is a great example of a minimalistic algorithm, both adhering to the same **PipelineOp** contract.


```python
from lib.pipeline_ops import *
from lib.vehicle_detection_ops import *
from lib.lane_detection_ops import *
from lib.datasets import *

calibration_op = None
```

#### Project Functions

> **NOTE** I extracted and enhanced to my needs many of these methods provided to us in the classroom lessons.


```python
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
        else: feature_image = np.copy(image)      

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
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
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
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
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
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
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
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
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
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def plot_histogram(image, nbins=32, bins_range=(0, 256), title=''):
    ch1h, ch2h, ch3h, bincen, feature_vec = ColorHistOp(image, nbins=nbins, bins_range=bins_range).perform().output()

    # Plot a figure with all three bar charts
    if ch1h is not None:
        fig = plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.bar(bincen, ch1h[0])
        plt.xlim(0, 256)
        plt.title(title+' Ch1 Histogram')
        plt.subplot(132)
        plt.bar(bincen, ch2h[0])
        plt.xlim(0, 256)
        plt.title(title+' Ch2 Histogram')
        plt.subplot(133)
        plt.bar(bincen, ch3h[0])
        plt.xlim(0, 256)
        plt.title(title+' Ch3 Histogram')
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

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    X_pred = []
    
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        X_pred.append(features)
    #5) Predict car or notcar
    predictions = clf.predict(scaler.transform(np.array(X_pred)))
    for i, prediction in enumerate(predictions):
        if prediction == 1:
            on_windows.append(windows[i])

    #8) Return windows for positive detections
    return on_windows

class Params():
    def __init__(
        self, 
        colorspace='YCrBr',
        orient=9,
        pix_per_cell=4, 
        cell_per_block=4, 
        hog_channel='ALL',
        spatial_size=(32, 32),
        hist_bins=32,
        spatial_feat=True,
        hist_feat=True,
        hog_feat=True
    ):
        self.colorspace = colorspace # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = orient # typically between 6 and 12
        self.pix_per_cell = pix_per_cell # HOG pixels per cell
        self.cell_per_block = cell_per_block # HOG cells per block
        self.hog_channel = hog_channel # Can be 0, 1, 2, or "ALL"
        self.spatial_size = spatial_size # Spatial binning dimensions
        self.hist_bins = hist_bins # Number of histogram bins
        self.spatial_feat = spatial_feat # Spatial features on or off
        self.hist_feat = hist_feat # Histogram features on or off
        self.hog_feat = hog_feat  # HOG features on or off
```

---

### Camera Calibration

##### Carryover from Project 4

My entire camera calibration algorithm may be found inside of **[CameraCalibrationOp#perform](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/107239a1530cc3b1f56016483941b28e5b70c26a/lib/lane_detection_ops.py#L54-L68)**. I've also exposed the **[#undistort](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/107239a1530cc3b1f56016483941b28e5b70c26a/lib/lane_detection_ops.py#L70-L83)** method which will undistort any raw image (ideally images taken by that camera but no code was put in place to validate camera source of image though EXIF data would be perfect place to look first).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![calibration1.jpg](../camera_cal/undistorted/calibration1.jpg)


```python
def calibrate_camera():
    global calibration_op
    
    if calibration_op == None:
        # base edges - doesn't work for all images in camera_cal directory (i.e., 1, 4, 5)
        calibration_images=glob.glob('camera_cal/calibration*.jpg')

        # I will now inject this calibration_op instance later on 
        # into my pipeline principally used to undistort the 
        # raw image.
        calibration_op = CameraCalibrationOp(
            calibration_images=calibration_images, 
            x_inside_corners=9, 
            y_inside_corners=6
        ).perform()
    return calibration_op
```

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for all steps from here on out is contained in the **Pipeline Operations** and **Pipeline Functions** code cell at the beginning of this IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images from the small dataset (via the **dataset_size** parameter to **[CarNotCarsDatasetOp](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/819921a611b9db62d09f11dfbf356f6f486ce718/lib/datasets.py#L7-L59)** class) as seen in the next cell.

> **NOTE** Later on down this notebook, when I'm about to train my LinearSVC classifier, I actually instantiate the larger dataset via **CarsNotCarsDatasetOp(dataset_size='big').perform()**.


```python
ds = CarsNotCarsDatasetOp(dataset_size='small').perform()
cars = ds.cars()
notcars = ds.notcars()
print('    # Cars: ' + str(len(cars)))
print('# Not cars: ' + str(len(notcars)))
```

        # Cars: 1196
    # Not cars: 1125


Here is an example of one of each of the `vehicle` and `non-vehicle` classes:


```python
# Generate a random index to look at a car image
ind_cars = np.random.randint(0, len(cars))
c_image = mpimg.imread(cars[ind_cars])


plt.title("CAR - {} - {}".format(ind_cars,cars[ind_cars]))
plt.subplot(121)
plt.imshow(c_image)
plt.show()

# Generate a random index to look at a notcar image
ind_notcars = np.random.randint(0, len(notcars))
nc_image = mpimg.imread(notcars[ind_notcars])

plt.title("NOTCAR - {} - {}".format(ind_notcars, notcars[ind_notcars]))
plt.subplot(122)
plt.imshow(nc_image)
plt.show()
```


![png](output_13_0.png)



![png](output_13_1.png)


Here is an example of the color histogram of various color spaces for the `vehicle` class:


```python
plot_histogram(c_image, title='[C][RGB]')
plot_histogram(cv2.cvtColor(c_image, cv2.COLOR_RGB2YCrCb), title='[C][YCrCb]')
plot_histogram(cv2.cvtColor(c_image, cv2.COLOR_RGB2YUV), title='[C][YUV]')
plot_histogram(cv2.cvtColor(c_image, cv2.COLOR_RGB2LUV), title='[C][LUV]')
plot_histogram(cv2.cvtColor(c_image, cv2.COLOR_RGB2HLS), title='[C][HLS]')
plot_histogram(cv2.cvtColor(c_image, cv2.COLOR_RGB2HSV), title='[C][HSV]')
```


![png](output_15_0.png)



![png](output_15_1.png)



![png](output_15_2.png)



![png](output_15_3.png)



![png](output_15_4.png)



![png](output_15_5.png)


Here is an example of the color histogram of various color spaces for the `non-vehicle` class:


```python
plot_histogram(nc_image, title='[NC][RGB]')
plot_histogram(cv2.cvtColor(nc_image, cv2.COLOR_RGB2YCrCb), title='[NC][YCrCb]')
plot_histogram(cv2.cvtColor(nc_image, cv2.COLOR_RGB2YUV), title='[NC][YUV]')
plot_histogram(cv2.cvtColor(nc_image, cv2.COLOR_RGB2LUV), title='[NC][LUV]')
plot_histogram(cv2.cvtColor(nc_image, cv2.COLOR_RGB2HLS), title='[NC][HLS]')
plot_histogram(cv2.cvtColor(nc_image, cv2.COLOR_RGB2HSV), title='[NC][HSV]')
```


![png](output_17_0.png)



![png](output_17_1.png)



![png](output_17_2.png)



![png](output_17_3.png)



![png](output_17_4.png)



![png](output_17_5.png)


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


```python
def extract_and_visualize(orig, gray, orient=8, pix_per_cell=8, cell_per_block=2, transform_sqrt=False):
    # Call our function with vis=True to see an image output
    features, hog_image = HOGExtractorOp(
        gray, orient, 
        pix_per_cell, cell_per_block, 
        visualize=True,
        feature_vec=False, 
        transform_sqrt=transform_sqrt
    ).output()

    # Plot the examples
    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(orig)
    plt.title('Orig. - {}'.format(str(ind_cars)))
    plt.subplot(132)
    plt.imshow(gray, cmap='gray')
    plt.title('Gray - {}'.format(str(ind_cars)))
    plt.subplot(133)
    plt.imshow(hog_image, cmap='hot')
    plt.title('HOG Vis. {}'.format(str(ind_cars)))
    plt.tight_layout()


# c_ind_cars = np.random.randint(0, len(cars))
# nc_ind_cars = np.random.randint(0, len(notcars))

# Read in the image
# ind_cars = 1165
#c_image = mpimg.imread(cars[c_ind_cars])
#nc_image = mpimg.imread(notcars[nc_ind_cars])

for i in range(3):
    cspace = 'YCrCb'
#     cspace = 'YUV'
#     cspace = 'HSV'
#     cspace = 'HLS'
    c_gray = cv2.cvtColor(c_image, eval('cv2.COLOR_RGB2'+cspace))[:,:,i]
    nc_gray = cv2.cvtColor(nc_image, eval('cv2.COLOR_RGB2'+cspace))[:,:,i]


    # Define HOG parameters
    orient = 8
    pix_per_cell = 8
    cell_per_block = 2


    extract_and_visualize(c_image, c_gray, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, transform_sqrt=True)
    # extract_and_visualize(c_image, c_gray, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, transform_sqrt=False)
    
    extract_and_visualize(nc_image, nc_gray, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, transform_sqrt=True)
    # extract_and_visualize(nc_image, nc_gray, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, transform_sqrt=False)
```


![png](output_19_0.png)



![png](output_19_1.png)



![png](output_19_2.png)



![png](output_19_3.png)



![png](output_19_4.png)



![png](output_19_5.png)


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the one with the highest accuracy is ultimately what I chose for my model.

I ultimately landed on using the `YCrCb` color space and HOG parameters of `orientations=7`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` because they yeilded the higest accuracy against the test set during training. I also liked how uniform the Cr and Cb color channels were more uniformly distributed in cars and less so for non-cars compared to other color spaces.

Each item outlined in the table below is initiatlized and commented out in the cell below. Feel free to have a go at it.


|  colorspace | orient | pix_per_cell | cell_per_block | hog_channel  |   accuracy   |
|:--------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| YUV | 9 | 8 | 2 | 0 | 97% |
| YUV | 8 | 8 | 2 | 0 | 98% |
| YUV | 8 | 8 | 2 | ALL | 98.5% |
| YUV | 8 | 7 | 2 | ALL | 99% |
| YCrCb | 8 | 7 | 2 | ALL | 99-100% |
| YCrCb* | 7 | 8 | 2 | ALL | 99.24% |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM as explained in the next few cells using the following approach:

1. I instantiate `Params` with the HOG parameters that yielded the best overall prediction accuracy against the test set.

2. I then invoke `#extract_features` which extracts HOG features from each training image for both the cars and not cars datasets passing the same parameter to each.

3. I then stack both cars and not cars datasets into a single array called `X` which correspond to my training features.

4. Using sklearn.preprocessing.StandardScaler(), I normalize my feature vectors for training my classifier. 

5. Then I apply the same scaling to each of the feature vectors.

6. Next, I created my training labels by using np.hstack which assigns the label `1` for each item from the `cars` training set and the label `0` for each item in the `notcars` training set.

7. Then I split up the data into randomized 80% training and 20% test sets using `sklearn.model_selection.train_test_split`. This automatically shuffles my dataset.

8. Using `sklearn.svm.LinearSVC`, I fit my training features and labels to the model.

9. Finally, I run a prediction against my model and print some statistics to the console below the next Jupyter Notebook cell.


```python
################################################################################################################
# params = Params(colorspace='YUV', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0) #97%
# params = Params(colorspace='YUV', orient=8, pix_per_cell=8, cell_per_block=2, hog_channel=0) #98%
# params = Params(colorspace='YUV', orient=8, pix_per_cell=8, cell_per_block=2, hog_channel='ALL') #98.5%
# params = Params(colorspace='YUV', orient=8, pix_per_cell=7, cell_per_block=2, hog_channel='ALL') #99%
# params = Params(colorspace='YCrCb', orient=8, pix_per_cell=7, cell_per_block=2, hog_channel='ALL') #99-100%
# params = Params(colorspace='YCrCb', orient=7, pix_per_cell=4, cell_per_block=2, hog_channel='ALL') #99-100%
# params = Params(colorspace='YCrCb', orient=9, pix_per_cell=4, cell_per_block=4, hog_channel='ALL') #99-100%
params = Params(colorspace='YCrCb', orient=7, pix_per_cell=8, cell_per_block=2, hog_channel='ALL') #99-100%
################################################################################################################

t=time.time()
car_features = extract_features(cars, color_space=params.colorspace, orient=params.orient, 
                        pix_per_cell=params.pix_per_cell, cell_per_block=params.cell_per_block, 
                        hog_channel=params.hog_channel)
notcar_features = extract_features(notcars, color_space=params.colorspace, orient=params.orient, 
                        pix_per_cell=params.pix_per_cell, cell_per_block=params.cell_per_block, 
                        hog_channel=params.hog_channel)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',params.orient,'orientations',params.pix_per_cell,
    'pixels per cell and', params.cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC(C=0.01)
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = len(X_test)
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
```

    10.86 Seconds to extract HOG features...
    Using: 7 orientations 8 pixels per cell and 2 cells per block
    Feature vector length: 7284
    0.47 Seconds to train SVC...
    Test Accuracy of SVC =  0.9978
    My SVC predicts:  [ 1.  0.  1.  0.  0.  0.  1.  0.  0.  1.  1.  0.  0.  0.  1.  1.  0.  0.
      0.  1.  1.  1.  0.  1.  1.  0.  0.  1.  0.  1.  0.  1.  0.  0.  0.  0.
      1.  0.  0.  0.  1.  1.  1.  0.  1.  0.  1.  1.  1.  0.  1.  0.  1.  1.
      1.  1.  1.  1.  0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  0.  1.  1.  1.
      1.  1.  0.  1.  1.  0.  1.  1.  0.  0.  1.  0.  1.  1.  0.  0.  1.  0.
      1.  0.  1.  1.  0.  0.  0.  1.  0.  1.  0.  1.  0.  0.  0.  0.  0.  1.
      1.  1.  0.  1.  1.  1.  1.  1.  0.  1.  0.  1.  1.  1.  1.  1.  1.  1.
      1.  0.  0.  1.  1.  1.  0.  0.  0.  1.  0.  0.  0.  1.  0.  1.  0.  0.
      0.  1.  0.  1.  0.  0.  1.  1.  0.  1.  1.  1.  0.  1.  1.  1.  0.  0.
      1.  0.  0.  0.  0.  1.  1.  1.  0.  0.  1.  0.  1.  0.  0.  0.  1.  0.
      0.  0.  0.  1.  1.  0.  0.  1.  1.  0.  1.  0.  0.  0.  1.  1.  1.  0.
      0.  1.  1.  1.  0.  0.  0.  0.  1.  1.  1.  1.  0.  1.  1.  0.  1.  0.
      1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  0.  1.  0.  1.  0.  0.  1.  0.
      0.  1.  0.  1.  0.  0.  1.  1.  0.  0.  0.  1.  1.  0.  1.  1.  1.  1.
      1.  1.  0.  0.  0.  0.  1.  1.  0.  1.  0.  0.  1.  1.  0.  1.  1.  1.
      1.  1.  0.  1.  0.  0.  1.  0.  0.  0.  0.  0.  1.  0.  0.  0.  1.  1.
      1.  0.  0.  1.  1.  1.  1.  0.  0.  1.  0.  1.  1.  0.  0.  0.  1.  1.
      1.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  1.
      1.  1.  0.  0.  1.  0.  0.  0.  0.  0.  1.  1.  0.  1.  1.  0.  0.  0.
      1.  0.  1.  0.  0.  0.  1.  1.  1.  0.  0.  0.  0.  1.  0.  1.  1.  1.
      0.  1.  1.  0.  0.  0.  1.  1.  1.  0.  0.  1.  0.  1.  0.  0.  1.  0.
      0.  0.  1.  1.  1.  0.  0.  1.  1.  0.  0.  1.  0.  0.  0.  0.  0.  1.
      1.  0.  0.  0.  0.  0.  0.  1.  1.  1.  0.  1.  1.  1.  0.  0.  1.  0.
      1.  1.  1.  0.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  1.  0.  0.
      0.  0.  0.  1.  1.  0.  1.  1.  0.  0.  0.  1.  0.  1.  1.  1.  1.  1.
      1.  0.  1.  0.  0.  1.  0.  0.  1.  1.  0.  1.  1.  1.  1.]
    For these 465 labels:  [ 1.  0.  1.  0.  0.  0.  1.  0.  0.  1.  1.  0.  0.  0.  1.  1.  0.  0.
      0.  1.  1.  1.  0.  1.  1.  0.  0.  1.  0.  1.  0.  1.  0.  0.  0.  0.
      1.  0.  0.  0.  1.  1.  1.  0.  1.  0.  1.  1.  1.  0.  1.  0.  1.  1.
      1.  1.  1.  1.  0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  0.  1.  1.  1.
      1.  1.  0.  1.  1.  0.  1.  1.  0.  0.  1.  0.  1.  1.  0.  0.  1.  0.
      1.  0.  1.  1.  0.  0.  0.  1.  0.  1.  0.  1.  0.  0.  0.  0.  0.  1.
      1.  1.  0.  1.  1.  1.  1.  1.  0.  1.  0.  1.  1.  1.  1.  1.  1.  1.
      1.  0.  0.  1.  1.  1.  0.  0.  0.  1.  0.  0.  0.  1.  0.  1.  0.  0.
      0.  1.  0.  1.  0.  0.  1.  1.  0.  1.  1.  1.  0.  1.  1.  1.  0.  0.
      1.  0.  0.  0.  0.  1.  1.  1.  0.  0.  1.  0.  1.  0.  0.  0.  1.  0.
      0.  0.  0.  1.  1.  0.  0.  1.  1.  0.  1.  0.  0.  0.  1.  1.  1.  0.
      0.  1.  1.  1.  0.  0.  0.  0.  1.  1.  1.  1.  0.  1.  1.  0.  1.  0.
      1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  0.  1.  0.  1.  0.  0.  1.  0.
      0.  1.  0.  1.  0.  0.  1.  1.  0.  0.  0.  1.  1.  0.  1.  1.  1.  1.
      1.  1.  0.  0.  0.  0.  1.  1.  0.  1.  0.  0.  1.  1.  0.  1.  1.  1.
      1.  1.  0.  1.  0.  0.  1.  0.  0.  0.  0.  0.  1.  0.  0.  0.  1.  1.
      1.  0.  0.  1.  1.  1.  1.  0.  0.  1.  0.  1.  1.  0.  0.  0.  1.  1.
      1.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  1.
      1.  1.  0.  0.  1.  0.  0.  0.  0.  0.  1.  1.  0.  1.  1.  0.  0.  0.
      1.  0.  1.  0.  0.  0.  1.  1.  1.  0.  0.  0.  0.  1.  0.  1.  1.  1.
      0.  1.  1.  0.  0.  0.  1.  1.  1.  0.  0.  1.  0.  1.  0.  0.  1.  0.
      0.  0.  1.  1.  1.  0.  0.  1.  1.  0.  0.  1.  0.  0.  0.  0.  0.  1.
      1.  0.  0.  0.  0.  0.  0.  1.  0.  1.  0.  1.  1.  1.  0.  0.  1.  0.
      1.  1.  1.  0.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  1.  0.  0.
      0.  0.  0.  1.  1.  0.  1.  1.  0.  0.  0.  1.  0.  1.  1.  1.  1.  1.
      1.  0.  1.  0.  0.  1.  0.  0.  1.  1.  0.  1.  1.  1.  1.]
    0.01626 Seconds to predict 465 labels with SVC


---

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

##### Vehicle Detection Pipeline Choices Explained

I commented each section of my pipeline in **VehicleDetectionOp#perform** (declared below in the Video Implementation section). Here is the documented algorithm for your viewing pleasure which explains precisely how I implemented my sliding window search:

```python
def perform(self):
    result = self.__img
    if self.__img is not None:
        svc = self.__svc
        X_scaler = self.__X_scaler
        params = self.__params
        image = self.__img
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        image = image.astype(np.float32)/255
        # Build our collection of search windows.
        # Since I know I'm in the far left lane in the project video, 
        # I decided to scan only the center and right lanes for 
        # performance purposes.
        #
        # After trying out over 100 different window kernel size and 
        # overlap permutations, I discovered that having a very high 
        # overlap (i.e., >=0.7) yielded more possible detections and 
        # the greatest anomoly detection accuracy after applying 
        # the heat map.
        #
        # Having a higher overlap has its pros and cons so the below 
        # windows have a healthy balance of performance and accuracy. 
        #
        # I also noticed that cars are not actually square so sliding 
        # a square kernel across non-square objects introduces errors. 
        # So, after trying out 7-8 different aspect ratios, I settled 
        # on a simple 5:4 aspect ratio giving me slighty more accurate 
        # readings across each frame.
        #
        # If I had to take a guess right now, I'd say my detection 
        # algorithm is as accurate as my LinearSVC classifier (~99.24%). 
        # There are still a few frames where the algorithm misses its 
        # target but I am satisfied with my first pass and look forward 
        # to peer feedback for tips and tricks.
        #
        # In addition, introducing a Vehicle class which is essentially 
        # responsible for tracking a single vehicle over time will ensure 
        # that even if I miss-calculate one out of seven frame, I'd 
        # essentially keep a running average from the previous 6 frames 
        # to fall back to.
        overlap = 0.9
        windows = []
        # I start off with a single 64px tall window search space with 0.9 
        # overlap to slide a 120x96 kernel across the horizon to pick up 
        # vehicles farther away. I wanted to apply the highest overlap at 
        # the smaller kernel size in the smallest search space because it 
        # is more accurate but also takes a lot longer to execute.
        windows += slide_window(image, x_start_stop=[704, None], y_start_stop=[375, 439], xy_window=(int(96*1.25), 96), xy_overlap=(overlap, overlap)) 
        # Next, I define a 128px tall search space with essentially a 0.81 
        # overlap to slide a 120x96 kernel accross. This essentially adds a 
        # bit more confidence around any detections from the previous window 
        # search. This also means that it was necessary for me to increase 
        # my heatmap threshold to remove any anomolies around the actual 
        # vehicle (e.g., road signs).
        windows += slide_window(image, x_start_stop=[768, None], y_start_stop=[375, 567], xy_window=(int(96*1.25), 96), xy_overlap=(overlap*0.9, overlap*0.9))
        # Finally, I sweep a 280x224 kernel across the entire search space 
        # with the full 0.9 overlap to pick up vehicles that may be right 
        # next to me or just entering the frame (i.e., closer to me).
        windows += slide_window(image, x_start_stop=[768, None], y_start_stop=[375, 695], xy_window=(int(224*1.25), 224), xy_overlap=(overlap*1., overlap*1.))
        # With the optimal windows identified, let's predict whether there's 
        # a vehicle inside of each window.
        #
        # This is by far the most process intensive method so it's imperative 
        # we address all performance concerns prior to reaching this phase.
        # 
        # As it stands right now with the current windows, it take ~2-2.5s per 
        # frame.
        t=time.time()
        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=params.colorspace, 
                                spatial_size=params.spatial_size, hist_bins=params.hist_bins, 
                                orient=params.orient, pix_per_cell=params.pix_per_cell, 
                                cell_per_block=params.cell_per_block, 
                                hog_channel=params.hog_channel, spatial_feat=params.spatial_feat, 
                                hist_feat=params.hist_feat, hog_feat=params.hog_feat)                       
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to search and locate {} windows...'.format(len(hot_windows)))

        # visualize the detected windows if visualization is enabled
        if self.vis_detections:
            # Draw the detected windows on top of the original image.
            window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=3)
            PlotImageOp(window_img, title="Detected Vehicles").perform()

        # A clever algorithm I was about to implement prior to obtaining optimal detections 
        # with the current searchable windows was to actually track the heat map over 
        # time instead of averaging windows over time. I didnt' end up using it in the end 
        # but I am leaving my code in here in hopes I can experiment with it later on.
        #
        # The idea behind this algorithm is to start off with a base threshold and 
        # to continue to add +1 to new detections to the heat map for n frames then 
        # start from scratch after each nth frame (i.e., cool the heatmap down).
        base_thresh = 7
        if True: #self.heat == None or (self.current_frame%base_thresh) == 0:
            self.heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = self.heat
        # (UNUSED) have a more lenient theshold by decaying over time
        heat_thresh = (base_thresh+((base_thresh*(self.current_frame%base_thresh))*((self.current_frame%base_thresh)/base_thresh)))
        # Weed out anomolies by accepting detections where at least 
        # 5 windows were predicted to have a vehicle in it.
        heatmap = add_heat(heat, hot_windows)
        heatmap = apply_threshold(heatmap, 5)
        self.heat = heatmap
        labels = label(heatmap)
        print(labels[1], 'cars found')
        # visualize the labels if visualization is enabled
        if self.vis_labels:
            PlotImageOp(labels[0], cmap='gray').perform()
        # visualize the heatmap if visualization is enabled
        if self.vis_heat:
            final_map = np.clip(heatmap-2, 0, 255)
            PlotImageOp(final_map, cmap='hot').perform()
        # Draw the labels onto the original image
        result = draw_labeled_bboxes(np.copy(image*255), labels)
    return self._apply_output(result)
```

### Training final Pipeline LinearSVC Classifier

1. First I instantiated the *`big`* dataset with over 8000 car and not car samples.

2. Then I instantiate the Params class which is fed into the `#train_classifier` method declared in the next cell.

3. `#train_classifier` instantiates LinearSVC with a C of 0.01 allowing for some noise (I used [this Stats exchance answer](http://stats.stackexchange.com/a/31067) for inspiration). 

4. Finally, `#train_classifier` It returns the instance of LinearSVC and the StandardScaler used to predict against inside of the **VehicleDetectionOp#perform** method later on in this notebook.

> My final classifier resulted in 99.24% accurate against 20% of the training set and expects a scaled feature vector of size `7284` at prediction time.


```python
ds = CarsNotCarsDatasetOp(dataset_size='big').perform()
cars = ds.cars()
notcars = ds.notcars()

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 1000
c_random_idxs = np.random.randint(0,len(cars), sample_size)
nc_random_idxs = np.random.randint(0,len(notcars), sample_size)
test_cars = cars #np.array(cars)[c_random_idxs] # cars[0:c_sample_size]
test_notcars = notcars #np.array(notcars)[nc_random_idxs] # notcars[0:nc_sample_size]

print('    # Cars: ' + str(len(test_cars)))
print('# Not cars: ' + str(len(test_notcars)))
```

        # Cars: 8792
    # Not cars: 8968



```python
def train_classifier(params, test_cars, test_notcars, C=1.):
    t=time.time()
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
    print(round(t2-t, 2), 'Seconds to extract HOG features...')

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

    print('Using:',params.orient,'orientations',params.pix_per_cell,
        'pixels per cell and', params.cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    # svc = LinearSVC(C=1.2) # 98.76%
    svc = LinearSVC(C=C)
    # from sklearn.ensemble import AdaBoostClassifier
    # svc = AdaBoostClassifier(learning_rate=0.1, algorithm='SAMME.R', n_estimators=50) # 86%
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    return svc, X_scaler

# params = Params(
#     colorspace='YCrCb', 
#     orient=7, 
#     pix_per_cell=8, 
#     cell_per_block=2, 
#     hog_channel='ALL', 
#     spatial_size=(32, 32), 
#     hist_bins=32,
#     spatial_feat=True,
#     hist_feat=True,
#     hog_feat=True,
#     y_start_stop=[400, 656]
# ) #98.37%

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
) #99.21%

svc, X_scaler = train_classifier(params, test_cars, test_notcars, C=0.01)
```

    77.15 Seconds to extract HOG features...
    Using: 7 orientations 8 pixels per cell and 2 cells per block
    Feature vector length: 7284
    15.86 Seconds to train SVC...
    Test Accuracy of SVC =  0.9916


#### Sliding Window Performance Measurement Cell

I used this cell to fine-tune my sliding window algorithm and accepted a `<3s` performance hit per sample as it resulted in the fewest number of anomolies and a tighter window around the detected vehicles.


```python
images = []
# images = ['notes/bbox-example-image.jpg']
images += glob.glob('test_images/*.jpg')
# images += ['test_images/test1.jpg']

results = []
for image in images:
    image = mpimg.imread(image)
    #print(image.shape[0:2][::-1])
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255
    # array of windows to search for cars
    windows = []
    # create sliding windows
    t=time.time()
    overlap = 0.9
    windows += slide_window(image, x_start_stop=[704, None], y_start_stop=[375, 439], xy_window=(int(96*1.25), 96), xy_overlap=(overlap, overlap)) 
    windows += slide_window(image, x_start_stop=[768, None], y_start_stop=[375, 567], xy_window=(int(96*1.25), 96), xy_overlap=(overlap*0.9, overlap*0.9))
    windows += slide_window(image, x_start_stop=[768, None], y_start_stop=[375, 695], xy_window=(int(224*1.25), 224), xy_overlap=(overlap*1., overlap*1.))
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to aggregate {} sliding windows...'.format(len(windows)))
    # search the sliding windows
    t=time.time()
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=params.colorspace, 
                            spatial_size=params.spatial_size, hist_bins=params.hist_bins, 
                            orient=params.orient, pix_per_cell=params.pix_per_cell, 
                            cell_per_block=params.cell_per_block, 
                            hog_channel=params.hog_channel, spatial_feat=params.spatial_feat, 
                            hist_feat=params.hist_feat, hog_feat=params.hog_feat)                       
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to search and locate {} windows...'.format(len(hot_windows)))
    results.append((image, hot_windows))
```

    0.0 Seconds to aggregate 725 sliding windows...
    3.02 Seconds to search and locate 110 windows...
    0.0 Seconds to aggregate 725 sliding windows...
    2.77 Seconds to search and locate 0 windows...
    0.0 Seconds to aggregate 725 sliding windows...
    2.97 Seconds to search and locate 6 windows...
    0.0 Seconds to aggregate 725 sliding windows...
    3.05 Seconds to search and locate 115 windows...
    0.0 Seconds to aggregate 725 sliding windows...
    3.09 Seconds to search and locate 78 windows...
    0.0 Seconds to aggregate 725 sliding windows...
    2.95 Seconds to search and locate 96 windows...


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

The ultimate step I took to optimize the performane of the classifier was to build different windows with various sizes based on knowledge that cars will be smaller closer to horizon and larget closer to my vehicle. Then I custom tuned the overlap for each window to achieve respectable performance so I didn't spend several hours to process the project video. Anything under 3 seconds was good enough for to achieve accurate results.

Here are some example images:

![400_IN_FINAL.jpg](../output_images/project_video/400_IN_FINAL.jpg)

![600_IN_FINAL.jpg](../output_images/project_video/600_IN_FINAL.jpg)

![800_IN_FINAL.jpg](../output_images/project_video/800_IN_FINAL.jpg)

![900_IN_FINAL.jpg](../output_images/project_video/900_IN_FINAL.jpg)

![1100_IN_FINAL.jpg](../output_images/project_video/1100_IN_FINAL.jpg)

![1200_IN_FINAL.jpg](../output_images/project_video/1200_IN_FINAL.jpg)

![1260_IN_FINAL.jpg](../output_images/project_video/1260_IN_FINAL.jpg)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

##### YouTube Videos

Please use these as your baseline reveiew.

[Video of Vehicle and Advance Lane Detection](https://youtu.be/asIqUYuIkM0)

[Video of Vehicle Detection Only](https://youtu.be/jsrWyRCsjJo)

##### Vehicle and Adanced Lane Finding Detection Only


```python
from IPython.display import HTML
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format('project_video_final_with_advanced_lane_detection.mp4'))
```





<video width="960" height="540" controls>
  <source src="project_video_final_with_advanced_lane_detection.mp4">
</video>




##### Vehicle Detection Only


```python
from IPython.display import HTML
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format('project_video_final.mp4'))
```





<video width="960" height="540" controls>
  <source src="project_video_final.mp4">
</video>





```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

class VehicleDetectionOp(PipelineOp):
    def __init__(self, svc, X_scaler, params, vis=False):
        PipelineOp.__init__(self)
        self.__svc = svc
        self.__X_scaler = X_scaler
        self.__params = params
        self.__img = None
        self.current_frame = -1
        # visualization flags
        self.vis_detections = vis
        self.vis_labels = vis
        self.vis_heat = vis
    
    def detect_vehicles(self, image, current_frame):
        self.current_frame = current_frame
        self.__img = image
        return self.perform().output()
    
    def perform(self):
        result = self.__img
        if self.__img is not None:
            svc = self.__svc
            X_scaler = self.__X_scaler
            params = self.__params
            image = self.__img
            # Uncomment the following line if you extracted training
            # data from .png images (scaled 0 to 1 by mpimg) and the
            # image you are searching is a .jpg (scaled 0 to 255)
            image = image.astype(np.float32)/255
            # Build our collection of search windows.
            # Since I know I'm in the far left lane in the project video, 
            # I decided to scan only the center and right lanes for 
            # performance purposes.
            #
            # After trying out over 100 different window kernel size and 
            # overlap permutations, I discovered that having a very high 
            # overlap (i.e., >=0.7) yielded more possible detections and 
            # the greatest anomoly detection accuracy after applying 
            # the heat map.
            #
            # Having a higher overlap has its pros and cons so the below 
            # windows have a healthy balance of performance and accuracy. 
            #
            # I also noticed that cars are not actually square so sliding 
            # a square kernel across non-square objects introduces errors. 
            # So, after trying out 7-8 different aspect ratios, I settled 
            # on a simple 5:4 aspect ratio giving me slighty more accurate 
            # readings across each frame.
            #
            # If I had to take aguess right now, I'd say my detection 
            # algorithm is as accurate as my LinearSVC classifier (~99.21%). 
            # There are still a few frames where the algorithm misses its 
            # target but I am satisfied with my first pass and look forward 
            # to peer feedback for tips and tricks.
            #
            # In addition, introducing a Vehicle class which is essentially 
            # responsible for tracking a single vehicle over time will ensure 
            # that even if I miss-calculate one our of seven frame, I'd 
            # essentially keep a running average from the previous 6 frames 
            # to fall back to.
            overlap = 0.9
            windows = []
            # I start off with a single 64px tall window search space with 0.9 
            # overlap to slide a 120x96 kernel across the horizon to pick up 
            # vehicles farther away. I wanted to apply the highest overlap at 
            # the smaller kernel size in the smallest search space because it 
            # is more accurate but also takes a lot longer to execute.
            windows += slide_window(image, x_start_stop=[704, None], y_start_stop=[375, 439], xy_window=(int(96*1.25), 96), xy_overlap=(overlap, overlap)) 
            # Next, I define a 128px tall search space with essentially a 0.81 
            # overlap to slide a 120x96 kernel accross. This essentially adds a 
            # bit more confidence around any detections from the previous window 
            # search. This also means that it was necessary for me to increase 
            # my heatmap threshold to remove any anomolies around the actual 
            # vehicle (e.g., road signs).
            windows += slide_window(image, x_start_stop=[768, None], y_start_stop=[375, 567], xy_window=(int(96*1.25), 96), xy_overlap=(overlap*0.9, overlap*0.9))
            # Finally, I sweep a 280x224 kernel across the entire search space 
            # with the full 0.9 overlap to pick up vehicles that may be right 
            # next to me or just entering the frame (i.e., closer to me).
            windows += slide_window(image, x_start_stop=[768, None], y_start_stop=[375, 695], xy_window=(int(224*1.25), 224), xy_overlap=(overlap*1., overlap*1.))
            # With the optimal windows identified, let's predict whether there's 
            # a vehicle inside of each window.
            #
            # This is by far the most process intensive method so it's imperative 
            # we address all performance concerns prior to reaching this phase.
            # 
            # As it stands right now with the current windows, it take ~2-2.5s per 
            # frame.
            t=time.time()
            hot_windows = search_windows(image, windows, svc, X_scaler, color_space=params.colorspace, 
                                    spatial_size=params.spatial_size, hist_bins=params.hist_bins, 
                                    orient=params.orient, pix_per_cell=params.pix_per_cell, 
                                    cell_per_block=params.cell_per_block, 
                                    hog_channel=params.hog_channel, spatial_feat=params.spatial_feat, 
                                    hist_feat=params.hist_feat, hog_feat=params.hog_feat)                       
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to search and locate {} windows...'.format(len(hot_windows)))

            # visualize the detected windows if visualization is enabled
            if self.vis_detections:
                # Draw the detected windows on top of the original image.
                window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=3)
                PlotImageOp(window_img, title="Detected Vehicles").perform()

            # A clever algorithm I was about to implement prior to obtaining optimal detections 
            # with the current searchable windows was to actually track the heat map over 
            # time instead of averaging windows over time. I didnt' end up using it in the end 
            # but I am leaving my code in here in hopes I can experiment with it later on.
            #
            # The idea behind this algorithm is to start off with a base threshold and 
            # to continue to add +1 to new detections to the heat map for n frames then 
            # start from scratch after each nth frame (i.e., cool the heatmap down).
            base_thresh = 7
            if True: #self.heat == None or (self.current_frame%base_thresh) == 0:
                self.heat = np.zeros_like(image[:,:,0]).astype(np.float)
            heat = self.heat

            # have a more lenient theshold by decaying over time
            heat_thresh = (base_thresh+((base_thresh*(self.current_frame%base_thresh))*((self.current_frame%base_thresh)/base_thresh)))

            # Weed out anomolies by excepting detections where at least 
            # 5 windows were predicted to have a vehicle in it.
            heatmap = add_heat(heat, hot_windows)
            heatmap = apply_threshold(heatmap, 5)
            self.heat = heatmap
            labels = label(heatmap)
            print(labels[1], 'cars found')
            
            # visualize the heatmap if visualization is enabled
            if self.vis_labels:
                PlotImageOp(labels[0], cmap='gray').perform()
            
            if self.vis_heat:
                final_map = np.clip(heatmap-2, 0, 255)
                PlotImageOp(final_map, cmap='hot').perform()

            # Draw the labels onto the original image
            result = draw_labeled_bboxes(np.copy(image*255), labels)
        return self._apply_output(result)
        

class PipelineRunner:
    def __init__(self, lane_assist_op, vehicle_detection_op, detect_lane=True, processed_images_save_dir=None):
        # Lane Detection operation algorithm used in #process_image
        self.lane_assist_op = lane_assist_op
        # Vehicle Detection operation algorithm used in #process_image
        self.vehicle_detection_op = vehicle_detection_op
        # used as a reference when saving images or leveraging windowed algorithms
        self.current_frame = -1
        # Flag indicator whether to draw detected lane surface onto final image
        self.detect_lane = detect_lane
        # Subdirector of /processed_images (an unversioned directory for all processed images)
        self.__processed_images_save_dir = processed_images_save_dir
        # Our pre-trained LinearSVC classifier used to predict against
        self.svc = svc
        # Scaler used to compute final feature vectors fet into classifier at prediction time
        self.X_scaler = X_scaler
        # Unused - Heat map used to track detected vehicles over a period of time (frames)
        self.heat = None
        
    def process_video(self, src_video_path, dst_video_path, audio=False):
        self.current_frame = -1
        # ensures all saved images for this video are created inside a subfolder 
        # corresponding to the video name
        self.__processed_images_save_dir = os.path.basename(src_video_path).split('.')[0]+'/'
        # Call our #process_image method for each frame 
        VideoFileClip(src_video_path).fl_image(self.process_image).write_videofile(dst_video_path, audio=audio)
    
    def process_image(self, image):
        self.current_frame += 1
        # save IN if we got it
        self.__save_image(image, 'IN')
        # Detect lane
        if self.detect_lane == True:
            subdir = '{}{}'.format(self.__processed_images_save_dir, self.current_frame)
            image = self.lane_assist_op.process_image(image, subdir).output()        
        # Detect vehicles
        image = self.vehicle_detection_op.detect_vehicles(image, self.current_frame)
        # Save final if we got it
        self.__save_image(image, 'OUT')
        return image
    
    def __save_image(self, image, name):
        if self.__processed_images_save_dir != None:
            cv2.imwrite('processed_images/{}{}_{}.jpg'.format(self.__processed_images_save_dir, self.current_frame, name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# Calibrates our camera and sets the `calibration_op` global 
# variable used by the LaneDetectionOp
calibrate_camera()
# Lane detection op (from P4) used by PipelineRunner
lane_assist_op = LaneDetectionOp(
    calibration_op,
    margin=100,
    kernel_size=15,
    sobelx_thresh=(20,100),
    sobely_thresh=(20,100),
    mag_grad_thresh=(20,250),
    dir_grad_thresh=(0.3, 1.3),
    color_space='HSV',
    color_channel=2
)
# Vehicle detection operation used by PipelineRunner
vehicle_detection_op = VehicleDetectionOp(
    svc, 
    X_scaler, 
    params,
    vis=False
)
# PipelineRunner which reads in a single image or source video and detects the 
# lane (when detect_lane=True) and vehicles in each video frame.
pipeline = PipelineRunner(
    lane_assist_op, 
    vehicle_detection_op, 
    detect_lane=True, 
    processed_images_save_dir='samples'
)
# See how well my pipeline performs against all .jpg images inside test_images directory
if False:
    images = []
#     images += ['notes/bbox-example-image.jpg']
#     images += glob.glob('test_images/*.jpg')
#     images += glob.glob('test_images/test3.jpg')
    for img_path in images:
        result = pipeline.process_image(mpimg.imread(img_path))
        PlotImageOp(result*255, title="{} - FINAL".format(img_path), cmap=None).perform()
# Run pipeline against the test_video.mp4
if False:
    pipeline.process_video('test_video.mp4', 'test_video_final.mp4')
# Run pipeline against the main project_video.mp4
if False:
    pipeline.process_video('project_video.mp4', 'project_video_final.mp4')
```

    2.93 Seconds to search and locate 1 windows...
    0 cars found
    [MoviePy] >>>> Building video project_video_final.mp4
    [MoviePy] Writing video project_video_final.mp4


      0%|          | 1/1261 [00:04<1:41:22,  4.83s/it]

    3.47 Seconds to search and locate 1 windows...
    0 cars found


      0%|          | 2/1261 [00:09<1:40:53,  4.81s/it]

    3.39 Seconds to search and locate 1 windows...
    0 cars found


      0%|          | 3/1261 [00:13<1:37:40,  4.66s/it]

    3.14 Seconds to search and locate 2 windows...
    0 cars found


      0%|          | 4/1261 [00:18<1:34:37,  4.52s/it]

    3.17 Seconds to search and locate 1 windows...
    0 cars found


      0%|          | 5/1261 [00:22<1:35:17,  4.55s/it]

    3.3 Seconds to search and locate 1 windows...
    0 cars found


      0%|          | 6/1261 [00:27<1:34:48,  4.53s/it]

    3.24 Seconds to search and locate 2 windows...
    0 cars found


      1%|          | 7/1261 [00:31<1:30:11,  4.32s/it]

    2.7 Seconds to search and locate 1 windows...
    0 cars found


      1%|          | 8/1261 [00:35<1:30:41,  4.34s/it]

    3.27 Seconds to search and locate 0 windows...
    0 cars found


      1%|          | 9/1261 [00:39<1:28:45,  4.25s/it]

    2.94 Seconds to search and locate 0 windows...
    0 cars found


      1%|          | 10/1261 [00:43<1:26:54,  4.17s/it]

    2.92 Seconds to search and locate 0 windows...
    0 cars found


      1%|          | 11/1261 [00:48<1:29:53,  4.32s/it]

    3.43 Seconds to search and locate 0 windows...
    0 cars found


      1%|          | 12/1261 [00:51<1:27:05,  4.18s/it]

    2.69 Seconds to search and locate 0 windows...
    0 cars found


      1%|          | 13/1261 [00:55<1:23:05,  3.99s/it]

    2.56 Seconds to search and locate 0 windows...
    0 cars found


      1%|          | 14/1261 [00:59<1:23:08,  4.00s/it]

    2.92 Seconds to search and locate 2 windows...
    0 cars found


      1%|          | 15/1261 [01:03<1:21:38,  3.93s/it]

    2.75 Seconds to search and locate 0 windows...
    0 cars found


      1%|▏         | 16/1261 [01:07<1:20:39,  3.89s/it]

    2.76 Seconds to search and locate 0 windows...
    0 cars found


      1%|▏         | 17/1261 [01:10<1:19:29,  3.83s/it]

    2.67 Seconds to search and locate 3 windows...
    0 cars found


      1%|▏         | 18/1261 [01:14<1:18:54,  3.81s/it]

    2.68 Seconds to search and locate 3 windows...
    0 cars found


      2%|▏         | 19/1261 [01:18<1:18:46,  3.81s/it]

    2.76 Seconds to search and locate 3 windows...
    0 cars found


      2%|▏         | 20/1261 [01:22<1:19:00,  3.82s/it]

    2.81 Seconds to search and locate 0 windows...
    0 cars found


      2%|▏         | 21/1261 [01:25<1:18:35,  3.80s/it]

    2.77 Seconds to search and locate 0 windows...
    0 cars found


      2%|▏         | 22/1261 [01:29<1:19:07,  3.83s/it]

    2.75 Seconds to search and locate 0 windows...
    0 cars found


      2%|▏         | 23/1261 [01:33<1:18:46,  3.82s/it]

    2.75 Seconds to search and locate 0 windows...
    0 cars found


      2%|▏         | 24/1261 [01:37<1:18:07,  3.79s/it]

    2.7 Seconds to search and locate 0 windows...
    0 cars found


      2%|▏         | 25/1261 [01:41<1:18:14,  3.80s/it]

    2.74 Seconds to search and locate 1 windows...
    0 cars found


      2%|▏         | 26/1261 [01:44<1:17:28,  3.76s/it]

    2.59 Seconds to search and locate 1 windows...
    0 cars found


      2%|▏         | 27/1261 [01:48<1:17:22,  3.76s/it]

    2.72 Seconds to search and locate 1 windows...
    0 cars found


      2%|▏         | 28/1261 [01:52<1:17:41,  3.78s/it]

    2.81 Seconds to search and locate 1 windows...
    0 cars found


      2%|▏         | 29/1261 [01:56<1:17:22,  3.77s/it]

    2.73 Seconds to search and locate 1 windows...
    0 cars found


      2%|▏         | 30/1261 [02:00<1:18:14,  3.81s/it]

    2.85 Seconds to search and locate 3 windows...
    0 cars found


      2%|▏         | 31/1261 [02:04<1:20:12,  3.91s/it]

    3.04 Seconds to search and locate 2 windows...
    0 cars found


      3%|▎         | 32/1261 [02:08<1:25:11,  4.16s/it]

    3.29 Seconds to search and locate 2 windows...
    0 cars found


      3%|▎         | 33/1261 [02:12<1:23:49,  4.10s/it]

    2.76 Seconds to search and locate 0 windows...
    0 cars found


      3%|▎         | 34/1261 [02:16<1:21:58,  4.01s/it]

    2.76 Seconds to search and locate 0 windows...
    0 cars found


      3%|▎         | 35/1261 [02:20<1:20:40,  3.95s/it]

    2.73 Seconds to search and locate 0 windows...
    0 cars found


      3%|▎         | 36/1261 [02:24<1:18:57,  3.87s/it]

    2.68 Seconds to search and locate 1 windows...
    0 cars found


      3%|▎         | 37/1261 [02:28<1:18:16,  3.84s/it]

    2.74 Seconds to search and locate 1 windows...
    0 cars found


      3%|▎         | 38/1261 [02:31<1:18:05,  3.83s/it]

    2.79 Seconds to search and locate 0 windows...
    0 cars found


      3%|▎         | 39/1261 [02:35<1:17:01,  3.78s/it]

    2.65 Seconds to search and locate 3 windows...
    0 cars found


      3%|▎         | 40/1261 [02:39<1:16:39,  3.77s/it]

    2.73 Seconds to search and locate 3 windows...
    0 cars found


      3%|▎         | 41/1261 [02:43<1:16:55,  3.78s/it]

    2.83 Seconds to search and locate 5 windows...
    0 cars found


      3%|▎         | 42/1261 [02:47<1:20:15,  3.95s/it]

    3.0 Seconds to search and locate 2 windows...
    0 cars found


      3%|▎         | 43/1261 [02:50<1:17:59,  3.84s/it]

    2.57 Seconds to search and locate 3 windows...
    0 cars found


      3%|▎         | 44/1261 [02:54<1:17:50,  3.84s/it]

    2.75 Seconds to search and locate 3 windows...
    0 cars found


      4%|▎         | 45/1261 [02:58<1:17:09,  3.81s/it]

    2.74 Seconds to search and locate 1 windows...
    0 cars found


      4%|▎         | 46/1261 [03:02<1:16:49,  3.79s/it]

    2.76 Seconds to search and locate 0 windows...
    0 cars found


      4%|▎         | 47/1261 [03:06<1:21:59,  4.05s/it]

    3.34 Seconds to search and locate 0 windows...
    0 cars found


      4%|▍         | 48/1261 [03:11<1:23:33,  4.13s/it]

    2.99 Seconds to search and locate 0 windows...
    0 cars found


      4%|▍         | 49/1261 [03:15<1:25:10,  4.22s/it]

    3.32 Seconds to search and locate 0 windows...
    0 cars found


      4%|▍         | 50/1261 [03:19<1:25:26,  4.23s/it]

    2.97 Seconds to search and locate 0 windows...
    0 cars found


      4%|▍         | 51/1261 [03:24<1:28:57,  4.41s/it]

    3.42 Seconds to search and locate 1 windows...
    0 cars found


      4%|▍         | 52/1261 [03:28<1:24:53,  4.21s/it]

    2.73 Seconds to search and locate 0 windows...
    0 cars found


      4%|▍         | 53/1261 [03:32<1:21:30,  4.05s/it]

    2.66 Seconds to search and locate 0 windows...
    0 cars found


      4%|▍         | 54/1261 [03:35<1:18:26,  3.90s/it]

    2.55 Seconds to search and locate 1 windows...
    0 cars found


      4%|▍         | 55/1261 [03:39<1:17:09,  3.84s/it]

    2.64 Seconds to search and locate 3 windows...
    0 cars found


      4%|▍         | 56/1261 [03:43<1:16:10,  3.79s/it]

    2.72 Seconds to search and locate 2 windows...
    0 cars found


      5%|▍         | 57/1261 [03:46<1:14:54,  3.73s/it]

    2.62 Seconds to search and locate 3 windows...
    0 cars found


      5%|▍         | 58/1261 [03:50<1:14:08,  3.70s/it]

    2.57 Seconds to search and locate 0 windows...
    0 cars found


      5%|▍         | 59/1261 [03:53<1:13:44,  3.68s/it]

    2.62 Seconds to search and locate 1 windows...
    0 cars found


      5%|▍         | 60/1261 [03:57<1:13:57,  3.70s/it]

    2.75 Seconds to search and locate 0 windows...
    0 cars found


      5%|▍         | 61/1261 [04:01<1:16:30,  3.83s/it]

    3.14 Seconds to search and locate 1 windows...
    0 cars found


      5%|▍         | 62/1261 [04:09<1:39:56,  5.00s/it]

    6.58 Seconds to search and locate 0 windows...
    0 cars found


      5%|▍         | 63/1261 [04:14<1:39:42,  4.99s/it]

    3.8 Seconds to search and locate 0 windows...
    0 cars found


      5%|▌         | 64/1261 [04:18<1:33:37,  4.69s/it]

    2.87 Seconds to search and locate 0 windows...
    0 cars found


      5%|▌         | 65/1261 [04:22<1:28:58,  4.46s/it]

    2.87 Seconds to search and locate 0 windows...
    0 cars found


      5%|▌         | 66/1261 [04:26<1:27:00,  4.37s/it]

    3.09 Seconds to search and locate 0 windows...
    0 cars found


      5%|▌         | 67/1261 [04:30<1:24:02,  4.22s/it]

    2.84 Seconds to search and locate 2 windows...
    0 cars found


      5%|▌         | 68/1261 [04:34<1:21:52,  4.12s/it]

    2.64 Seconds to search and locate 2 windows...
    0 cars found


      5%|▌         | 69/1261 [04:38<1:19:10,  3.99s/it]

    2.69 Seconds to search and locate 1 windows...
    0 cars found


      6%|▌         | 70/1261 [04:41<1:18:05,  3.93s/it]

    2.76 Seconds to search and locate 3 windows...
    0 cars found


      6%|▌         | 71/1261 [04:45<1:15:45,  3.82s/it]

    2.58 Seconds to search and locate 0 windows...
    0 cars found


      6%|▌         | 72/1261 [04:49<1:14:50,  3.78s/it]

    2.65 Seconds to search and locate 0 windows...
    0 cars found


      6%|▌         | 73/1261 [04:52<1:13:51,  3.73s/it]

    2.65 Seconds to search and locate 0 windows...
    0 cars found


      6%|▌         | 74/1261 [04:56<1:12:31,  3.67s/it]

    2.55 Seconds to search and locate 0 windows...
    0 cars found


      6%|▌         | 75/1261 [04:59<1:12:31,  3.67s/it]

    2.62 Seconds to search and locate 0 windows...
    0 cars found


      6%|▌         | 76/1261 [05:03<1:12:22,  3.66s/it]

    2.67 Seconds to search and locate 0 windows...
    0 cars found


      6%|▌         | 77/1261 [05:07<1:12:17,  3.66s/it]

    2.68 Seconds to search and locate 0 windows...
    0 cars found


      6%|▌         | 78/1261 [05:10<1:11:15,  3.61s/it]

    2.53 Seconds to search and locate 0 windows...
    0 cars found


      6%|▋         | 79/1261 [05:14<1:10:57,  3.60s/it]

    2.55 Seconds to search and locate 1 windows...
    0 cars found


      6%|▋         | 80/1261 [05:17<1:11:13,  3.62s/it]

    2.69 Seconds to search and locate 4 windows...
    0 cars found


      6%|▋         | 81/1261 [05:21<1:11:03,  3.61s/it]

    2.62 Seconds to search and locate 2 windows...
    0 cars found


      7%|▋         | 82/1261 [05:25<1:10:33,  3.59s/it]

    2.55 Seconds to search and locate 4 windows...
    0 cars found


      7%|▋         | 83/1261 [05:28<1:10:50,  3.61s/it]

    2.68 Seconds to search and locate 1 windows...
    0 cars found


      7%|▋         | 84/1261 [05:32<1:10:58,  3.62s/it]

    2.66 Seconds to search and locate 0 windows...
    0 cars found


      7%|▋         | 85/1261 [05:35<1:10:20,  3.59s/it]

    2.55 Seconds to search and locate 0 windows...
    0 cars found


      7%|▋         | 86/1261 [05:39<1:10:38,  3.61s/it]

    2.62 Seconds to search and locate 0 windows...
    0 cars found


      7%|▋         | 87/1261 [05:43<1:10:29,  3.60s/it]

    2.63 Seconds to search and locate 1 windows...
    0 cars found


      7%|▋         | 88/1261 [05:46<1:10:22,  3.60s/it]

    2.63 Seconds to search and locate 0 windows...
    0 cars found


      7%|▋         | 89/1261 [05:50<1:10:07,  3.59s/it]

    2.56 Seconds to search and locate 0 windows...
    0 cars found


      7%|▋         | 90/1261 [05:53<1:10:09,  3.59s/it]

    2.63 Seconds to search and locate 0 windows...
    0 cars found


      7%|▋         | 91/1261 [05:57<1:10:20,  3.61s/it]

    2.65 Seconds to search and locate 0 windows...
    0 cars found


      7%|▋         | 92/1261 [06:01<1:09:58,  3.59s/it]

    2.59 Seconds to search and locate 1 windows...
    0 cars found


      7%|▋         | 93/1261 [06:04<1:09:57,  3.59s/it]

    2.56 Seconds to search and locate 2 windows...
    0 cars found


      7%|▋         | 94/1261 [06:08<1:09:54,  3.59s/it]

    2.65 Seconds to search and locate 2 windows...
    0 cars found


      8%|▊         | 95/1261 [06:11<1:10:00,  3.60s/it]

    2.64 Seconds to search and locate 1 windows...
    0 cars found


      8%|▊         | 96/1261 [06:15<1:09:17,  3.57s/it]

    2.52 Seconds to search and locate 0 windows...
    0 cars found


      8%|▊         | 97/1261 [06:19<1:09:33,  3.59s/it]

    2.63 Seconds to search and locate 1 windows...
    0 cars found


      8%|▊         | 98/1261 [06:22<1:09:57,  3.61s/it]

    2.66 Seconds to search and locate 0 windows...
    0 cars found


      8%|▊         | 99/1261 [06:26<1:09:41,  3.60s/it]

    2.59 Seconds to search and locate 0 windows...
    0 cars found


      8%|▊         | 100/1261 [06:29<1:09:43,  3.60s/it]

    2.59 Seconds to search and locate 0 windows...
    0 cars found


      8%|▊         | 101/1261 [06:33<1:09:50,  3.61s/it]

    2.64 Seconds to search and locate 0 windows...
    0 cars found


      8%|▊         | 102/1261 [06:37<1:10:00,  3.62s/it]

    2.68 Seconds to search and locate 0 windows...
    0 cars found


      8%|▊         | 103/1261 [06:40<1:09:48,  3.62s/it]

    2.61 Seconds to search and locate 0 windows...
    0 cars found


      8%|▊         | 104/1261 [06:44<1:09:37,  3.61s/it]

    2.57 Seconds to search and locate 1 windows...
    0 cars found


      8%|▊         | 105/1261 [06:48<1:09:43,  3.62s/it]

    2.67 Seconds to search and locate 2 windows...
    0 cars found


      8%|▊         | 106/1261 [06:51<1:09:16,  3.60s/it]

    2.57 Seconds to search and locate 3 windows...
    0 cars found


      8%|▊         | 107/1261 [06:55<1:09:01,  3.59s/it]

    2.57 Seconds to search and locate 3 windows...
    0 cars found


      9%|▊         | 108/1261 [06:58<1:09:31,  3.62s/it]

    2.7 Seconds to search and locate 2 windows...
    0 cars found


      9%|▊         | 109/1261 [07:02<1:10:22,  3.67s/it]

    2.79 Seconds to search and locate 0 windows...
    0 cars found


      9%|▊         | 110/1261 [07:06<1:10:52,  3.69s/it]

    2.76 Seconds to search and locate 0 windows...
    0 cars found


      9%|▉         | 111/1261 [07:10<1:13:10,  3.82s/it]

    3.05 Seconds to search and locate 1 windows...
    0 cars found


      9%|▉         | 112/1261 [07:14<1:12:39,  3.79s/it]

    2.67 Seconds to search and locate 1 windows...
    0 cars found


      9%|▉         | 113/1261 [07:17<1:12:07,  3.77s/it]

    2.72 Seconds to search and locate 1 windows...
    0 cars found


      9%|▉         | 114/1261 [07:21<1:11:01,  3.72s/it]

    2.62 Seconds to search and locate 2 windows...
    0 cars found


      9%|▉         | 115/1261 [07:25<1:10:29,  3.69s/it]

    2.62 Seconds to search and locate 3 windows...
    0 cars found


      9%|▉         | 116/1261 [07:28<1:10:31,  3.70s/it]

    2.72 Seconds to search and locate 3 windows...
    0 cars found


      9%|▉         | 117/1261 [07:32<1:12:23,  3.80s/it]

    2.96 Seconds to search and locate 3 windows...
    0 cars found


      9%|▉         | 118/1261 [07:36<1:13:17,  3.85s/it]

    2.91 Seconds to search and locate 1 windows...
    0 cars found


      9%|▉         | 119/1261 [07:40<1:13:38,  3.87s/it]

    2.9 Seconds to search and locate 7 windows...
    0 cars found


     10%|▉         | 120/1261 [07:44<1:14:02,  3.89s/it]

    2.94 Seconds to search and locate 7 windows...
    0 cars found


     10%|▉         | 121/1261 [07:48<1:13:08,  3.85s/it]

    2.68 Seconds to search and locate 2 windows...
    0 cars found


     10%|▉         | 122/1261 [07:52<1:12:56,  3.84s/it]

    2.82 Seconds to search and locate 0 windows...
    0 cars found


     10%|▉         | 123/1261 [07:55<1:11:18,  3.76s/it]

    2.6 Seconds to search and locate 0 windows...
    0 cars found


     10%|▉         | 124/1261 [07:59<1:12:00,  3.80s/it]

    2.86 Seconds to search and locate 0 windows...
    0 cars found


     10%|▉         | 125/1261 [08:03<1:13:06,  3.86s/it]

    2.93 Seconds to search and locate 0 windows...
    0 cars found


     10%|▉         | 126/1261 [08:07<1:12:14,  3.82s/it]

    2.68 Seconds to search and locate 0 windows...
    0 cars found


     10%|█         | 127/1261 [08:11<1:11:03,  3.76s/it]

    2.63 Seconds to search and locate 0 windows...
    0 cars found


     10%|█         | 128/1261 [08:15<1:12:42,  3.85s/it]

    2.98 Seconds to search and locate 1 windows...
    0 cars found


     10%|█         | 129/1261 [08:18<1:12:17,  3.83s/it]

    2.81 Seconds to search and locate 0 windows...
    0 cars found


     10%|█         | 130/1261 [08:22<1:12:52,  3.87s/it]

    2.92 Seconds to search and locate 1 windows...
    0 cars found


     10%|█         | 131/1261 [08:26<1:13:44,  3.92s/it]

    2.92 Seconds to search and locate 3 windows...
    0 cars found


     10%|█         | 132/1261 [08:31<1:15:14,  4.00s/it]

    3.1 Seconds to search and locate 4 windows...
    0 cars found


     11%|█         | 133/1261 [08:34<1:13:16,  3.90s/it]

    2.61 Seconds to search and locate 2 windows...
    0 cars found


     11%|█         | 134/1261 [08:38<1:12:00,  3.83s/it]

    2.71 Seconds to search and locate 1 windows...
    0 cars found


     11%|█         | 135/1261 [08:42<1:12:05,  3.84s/it]

    2.86 Seconds to search and locate 1 windows...
    0 cars found


     11%|█         | 136/1261 [08:45<1:10:44,  3.77s/it]

    2.64 Seconds to search and locate 2 windows...
    0 cars found


     11%|█         | 137/1261 [08:49<1:11:20,  3.81s/it]

    2.85 Seconds to search and locate 0 windows...
    0 cars found


     11%|█         | 138/1261 [08:54<1:14:27,  3.98s/it]

    3.23 Seconds to search and locate 2 windows...
    0 cars found


     11%|█         | 139/1261 [08:58<1:14:40,  3.99s/it]

    2.95 Seconds to search and locate 1 windows...
    0 cars found


     11%|█         | 140/1261 [09:02<1:13:32,  3.94s/it]

    2.8 Seconds to search and locate 1 windows...
    0 cars found


     11%|█         | 141/1261 [09:05<1:11:46,  3.85s/it]

    2.58 Seconds to search and locate 0 windows...
    0 cars found


     11%|█▏        | 142/1261 [09:09<1:10:56,  3.80s/it]

    2.67 Seconds to search and locate 0 windows...
    0 cars found


     11%|█▏        | 143/1261 [09:13<1:10:01,  3.76s/it]

    2.67 Seconds to search and locate 3 windows...
    0 cars found


     11%|█▏        | 144/1261 [09:16<1:09:01,  3.71s/it]

    2.62 Seconds to search and locate 2 windows...
    0 cars found


     11%|█▏        | 145/1261 [09:20<1:08:32,  3.69s/it]

    2.6 Seconds to search and locate 3 windows...
    0 cars found


     12%|█▏        | 146/1261 [09:23<1:08:15,  3.67s/it]

    2.66 Seconds to search and locate 0 windows...
    0 cars found


     12%|█▏        | 147/1261 [09:27<1:08:18,  3.68s/it]

    2.71 Seconds to search and locate 2 windows...
    0 cars found


     12%|█▏        | 148/1261 [09:31<1:08:00,  3.67s/it]

    2.63 Seconds to search and locate 1 windows...
    0 cars found


     12%|█▏        | 149/1261 [09:34<1:07:47,  3.66s/it]

    2.57 Seconds to search and locate 1 windows...
    0 cars found


     12%|█▏        | 150/1261 [09:38<1:07:51,  3.66s/it]

    2.7 Seconds to search and locate 0 windows...
    0 cars found


     12%|█▏        | 151/1261 [09:42<1:07:44,  3.66s/it]

    2.67 Seconds to search and locate 1 windows...
    0 cars found


     12%|█▏        | 152/1261 [09:45<1:06:59,  3.62s/it]

    2.58 Seconds to search and locate 0 windows...
    0 cars found


     12%|█▏        | 153/1261 [09:49<1:07:05,  3.63s/it]

    2.65 Seconds to search and locate 3 windows...
    0 cars found


     12%|█▏        | 154/1261 [09:52<1:06:54,  3.63s/it]

    2.65 Seconds to search and locate 1 windows...
    0 cars found


     12%|█▏        | 155/1261 [09:56<1:06:45,  3.62s/it]

    2.63 Seconds to search and locate 3 windows...
    0 cars found


     12%|█▏        | 156/1261 [10:00<1:06:36,  3.62s/it]

    2.59 Seconds to search and locate 5 windows...
    0 cars found


     12%|█▏        | 157/1261 [10:03<1:06:48,  3.63s/it]

    2.69 Seconds to search and locate 5 windows...
    0 cars found


     13%|█▎        | 158/1261 [10:07<1:06:59,  3.64s/it]

    2.7 Seconds to search and locate 5 windows...
    0 cars found


     13%|█▎        | 159/1261 [10:11<1:06:17,  3.61s/it]

    2.55 Seconds to search and locate 3 windows...
    0 cars found


     13%|█▎        | 160/1261 [10:14<1:06:16,  3.61s/it]

    2.55 Seconds to search and locate 3 windows...
    0 cars found


     13%|█▎        | 161/1261 [10:18<1:06:33,  3.63s/it]

    2.69 Seconds to search and locate 8 windows...
    1 cars found


     13%|█▎        | 162/1261 [10:22<1:06:31,  3.63s/it]

    2.63 Seconds to search and locate 8 windows...
    1 cars found


     13%|█▎        | 163/1261 [10:25<1:06:06,  3.61s/it]

    2.57 Seconds to search and locate 7 windows...
    1 cars found


     13%|█▎        | 164/1261 [10:29<1:06:21,  3.63s/it]

    2.66 Seconds to search and locate 5 windows...
    0 cars found


     13%|█▎        | 165/1261 [10:32<1:06:35,  3.65s/it]

    2.67 Seconds to search and locate 5 windows...
    0 cars found


     13%|█▎        | 166/1261 [10:36<1:06:09,  3.63s/it]

    2.61 Seconds to search and locate 7 windows...
    0 cars found


     13%|█▎        | 167/1261 [10:40<1:07:07,  3.68s/it]

    2.76 Seconds to search and locate 10 windows...
    1 cars found


     13%|█▎        | 168/1261 [10:44<1:07:25,  3.70s/it]

    2.73 Seconds to search and locate 18 windows...
    1 cars found


     13%|█▎        | 169/1261 [10:47<1:07:39,  3.72s/it]

    2.77 Seconds to search and locate 22 windows...
    1 cars found


     13%|█▎        | 170/1261 [10:51<1:06:45,  3.67s/it]

    2.57 Seconds to search and locate 28 windows...
    2 cars found


     14%|█▎        | 171/1261 [10:54<1:06:17,  3.65s/it]

    2.55 Seconds to search and locate 26 windows...
    1 cars found


     14%|█▎        | 172/1261 [10:58<1:06:27,  3.66s/it]

    2.71 Seconds to search and locate 20 windows...
    1 cars found


     14%|█▎        | 173/1261 [11:02<1:06:28,  3.67s/it]

    2.7 Seconds to search and locate 23 windows...
    1 cars found


     14%|█▍        | 174/1261 [11:05<1:05:45,  3.63s/it]

    2.58 Seconds to search and locate 27 windows...
    1 cars found


     14%|█▍        | 175/1261 [11:09<1:05:59,  3.65s/it]

    2.65 Seconds to search and locate 24 windows...
    1 cars found


     14%|█▍        | 176/1261 [11:13<1:05:59,  3.65s/it]

    2.66 Seconds to search and locate 28 windows...
    1 cars found


     14%|█▍        | 177/1261 [11:16<1:05:43,  3.64s/it]

    2.63 Seconds to search and locate 23 windows...
    1 cars found


     14%|█▍        | 178/1261 [11:20<1:05:49,  3.65s/it]

    2.62 Seconds to search and locate 23 windows...
    1 cars found


     14%|█▍        | 179/1261 [11:24<1:06:22,  3.68s/it]

    2.72 Seconds to search and locate 21 windows...
    1 cars found


     14%|█▍        | 180/1261 [11:27<1:06:32,  3.69s/it]

    2.72 Seconds to search and locate 29 windows...
    1 cars found


     14%|█▍        | 181/1261 [11:31<1:06:23,  3.69s/it]

    2.66 Seconds to search and locate 18 windows...
    1 cars found


     14%|█▍        | 182/1261 [11:35<1:05:53,  3.66s/it]

    2.58 Seconds to search and locate 19 windows...
    1 cars found


     15%|█▍        | 183/1261 [11:38<1:05:48,  3.66s/it]

    2.69 Seconds to search and locate 22 windows...
    1 cars found


     15%|█▍        | 184/1261 [11:42<1:05:45,  3.66s/it]

    2.66 Seconds to search and locate 23 windows...
    1 cars found


     15%|█▍        | 185/1261 [11:46<1:05:17,  3.64s/it]

    2.59 Seconds to search and locate 24 windows...
    1 cars found


     15%|█▍        | 186/1261 [11:49<1:05:32,  3.66s/it]

    2.63 Seconds to search and locate 24 windows...
    1 cars found


     15%|█▍        | 187/1261 [11:53<1:05:26,  3.66s/it]

    2.67 Seconds to search and locate 23 windows...
    1 cars found


     15%|█▍        | 188/1261 [11:57<1:05:38,  3.67s/it]

    2.71 Seconds to search and locate 21 windows...
    1 cars found


     15%|█▍        | 189/1261 [12:00<1:05:18,  3.66s/it]

    2.63 Seconds to search and locate 16 windows...
    1 cars found


     15%|█▌        | 190/1261 [12:04<1:05:04,  3.65s/it]

    2.6 Seconds to search and locate 17 windows...
    1 cars found


     15%|█▌        | 191/1261 [12:08<1:05:06,  3.65s/it]

    2.68 Seconds to search and locate 16 windows...
    1 cars found


     15%|█▌        | 192/1261 [12:11<1:05:00,  3.65s/it]

    2.64 Seconds to search and locate 27 windows...
    1 cars found


     15%|█▌        | 193/1261 [12:15<1:04:36,  3.63s/it]

    2.59 Seconds to search and locate 31 windows...
    1 cars found


     15%|█▌        | 194/1261 [12:19<1:04:49,  3.65s/it]

    2.68 Seconds to search and locate 30 windows...
    1 cars found


     15%|█▌        | 195/1261 [12:22<1:04:45,  3.64s/it]

    2.66 Seconds to search and locate 31 windows...
    1 cars found


     16%|█▌        | 196/1261 [12:26<1:04:37,  3.64s/it]

    2.63 Seconds to search and locate 29 windows...
    1 cars found


     16%|█▌        | 197/1261 [12:30<1:04:58,  3.66s/it]

    2.66 Seconds to search and locate 28 windows...
    1 cars found


     16%|█▌        | 198/1261 [12:33<1:04:59,  3.67s/it]

    2.7 Seconds to search and locate 33 windows...
    1 cars found


     16%|█▌        | 199/1261 [12:37<1:05:07,  3.68s/it]

    2.72 Seconds to search and locate 35 windows...
    1 cars found


     16%|█▌        | 200/1261 [12:41<1:05:38,  3.71s/it]

    2.7 Seconds to search and locate 32 windows...
    1 cars found


     16%|█▌        | 201/1261 [12:45<1:05:58,  3.73s/it]

    2.68 Seconds to search and locate 32 windows...
    1 cars found


     16%|█▌        | 202/1261 [12:48<1:06:06,  3.75s/it]

    2.74 Seconds to search and locate 32 windows...
    1 cars found


     16%|█▌        | 203/1261 [12:52<1:05:44,  3.73s/it]

    2.69 Seconds to search and locate 39 windows...
    1 cars found


     16%|█▌        | 204/1261 [12:56<1:05:04,  3.69s/it]

    2.61 Seconds to search and locate 38 windows...
    1 cars found


     16%|█▋        | 205/1261 [12:59<1:05:24,  3.72s/it]

    2.67 Seconds to search and locate 43 windows...
    1 cars found


     16%|█▋        | 206/1261 [13:03<1:05:15,  3.71s/it]

    2.7 Seconds to search and locate 34 windows...
    1 cars found


     16%|█▋        | 207/1261 [13:07<1:05:07,  3.71s/it]

    2.71 Seconds to search and locate 33 windows...
    1 cars found


     16%|█▋        | 208/1261 [13:10<1:04:37,  3.68s/it]

    2.62 Seconds to search and locate 38 windows...
    1 cars found


     17%|█▋        | 209/1261 [13:14<1:04:38,  3.69s/it]

    2.64 Seconds to search and locate 45 windows...
    1 cars found


     17%|█▋        | 210/1261 [13:18<1:04:28,  3.68s/it]

    2.67 Seconds to search and locate 42 windows...
    1 cars found


     17%|█▋        | 211/1261 [13:21<1:04:10,  3.67s/it]

    2.63 Seconds to search and locate 37 windows...
    1 cars found


     17%|█▋        | 212/1261 [13:25<1:03:53,  3.65s/it]

    2.61 Seconds to search and locate 39 windows...
    1 cars found


     17%|█▋        | 213/1261 [13:29<1:04:12,  3.68s/it]

    2.7 Seconds to search and locate 41 windows...
    1 cars found


     17%|█▋        | 214/1261 [13:32<1:04:08,  3.68s/it]

    2.66 Seconds to search and locate 42 windows...
    1 cars found


     17%|█▋        | 215/1261 [13:36<1:03:46,  3.66s/it]

    2.64 Seconds to search and locate 42 windows...
    1 cars found


     17%|█▋        | 216/1261 [13:40<1:03:49,  3.66s/it]

    2.62 Seconds to search and locate 35 windows...
    1 cars found


     17%|█▋        | 217/1261 [13:43<1:03:52,  3.67s/it]

    2.68 Seconds to search and locate 48 windows...
    1 cars found


     17%|█▋        | 218/1261 [13:47<1:04:00,  3.68s/it]

    2.72 Seconds to search and locate 46 windows...
    1 cars found


     17%|█▋        | 219/1261 [13:51<1:03:23,  3.65s/it]

    2.58 Seconds to search and locate 45 windows...
    1 cars found


     17%|█▋        | 220/1261 [13:54<1:03:21,  3.65s/it]

    2.58 Seconds to search and locate 43 windows...
    1 cars found


     18%|█▊        | 221/1261 [13:58<1:03:31,  3.66s/it]

    2.69 Seconds to search and locate 50 windows...
    1 cars found


     18%|█▊        | 222/1261 [14:02<1:03:39,  3.68s/it]

    2.7 Seconds to search and locate 49 windows...
    1 cars found


     18%|█▊        | 223/1261 [14:05<1:03:15,  3.66s/it]

    2.61 Seconds to search and locate 50 windows...
    1 cars found


     18%|█▊        | 224/1261 [14:09<1:03:11,  3.66s/it]

    2.62 Seconds to search and locate 44 windows...
    1 cars found


     18%|█▊        | 225/1261 [14:13<1:03:10,  3.66s/it]

    2.68 Seconds to search and locate 50 windows...
    1 cars found


     18%|█▊        | 226/1261 [14:16<1:03:09,  3.66s/it]

    2.67 Seconds to search and locate 46 windows...
    1 cars found


     18%|█▊        | 227/1261 [14:20<1:02:59,  3.66s/it]

    2.58 Seconds to search and locate 40 windows...
    1 cars found


     18%|█▊        | 228/1261 [14:24<1:02:49,  3.65s/it]

    2.64 Seconds to search and locate 45 windows...
    1 cars found


     18%|█▊        | 229/1261 [14:27<1:03:15,  3.68s/it]

    2.73 Seconds to search and locate 39 windows...
    1 cars found


     18%|█▊        | 230/1261 [14:31<1:02:58,  3.67s/it]

    2.64 Seconds to search and locate 42 windows...
    1 cars found


     18%|█▊        | 231/1261 [14:35<1:02:53,  3.66s/it]

    2.61 Seconds to search and locate 41 windows...
    1 cars found


     18%|█▊        | 232/1261 [14:38<1:03:32,  3.71s/it]

    2.79 Seconds to search and locate 42 windows...
    1 cars found


     18%|█▊        | 233/1261 [14:42<1:03:55,  3.73s/it]

    2.77 Seconds to search and locate 46 windows...
    1 cars found


     19%|█▊        | 234/1261 [14:46<1:03:52,  3.73s/it]

    2.72 Seconds to search and locate 45 windows...
    1 cars found


     19%|█▊        | 235/1261 [14:50<1:03:49,  3.73s/it]

    2.62 Seconds to search and locate 39 windows...
    1 cars found


     19%|█▊        | 236/1261 [14:53<1:03:21,  3.71s/it]

    2.67 Seconds to search and locate 48 windows...
    1 cars found


     19%|█▉        | 237/1261 [14:57<1:03:12,  3.70s/it]

    2.7 Seconds to search and locate 45 windows...
    1 cars found


     19%|█▉        | 238/1261 [15:01<1:02:50,  3.69s/it]

    2.63 Seconds to search and locate 46 windows...
    1 cars found


     19%|█▉        | 239/1261 [15:04<1:02:42,  3.68s/it]

    2.59 Seconds to search and locate 33 windows...
    1 cars found


     19%|█▉        | 240/1261 [15:08<1:02:38,  3.68s/it]

    2.68 Seconds to search and locate 47 windows...
    1 cars found


     19%|█▉        | 241/1261 [15:12<1:02:29,  3.68s/it]

    2.67 Seconds to search and locate 31 windows...
    1 cars found


     19%|█▉        | 242/1261 [15:15<1:02:02,  3.65s/it]

    2.59 Seconds to search and locate 46 windows...
    1 cars found


     19%|█▉        | 243/1261 [15:19<1:02:12,  3.67s/it]

    2.67 Seconds to search and locate 45 windows...
    1 cars found


     19%|█▉        | 244/1261 [15:23<1:02:02,  3.66s/it]

    2.66 Seconds to search and locate 49 windows...
    1 cars found


     19%|█▉        | 245/1261 [15:26<1:01:47,  3.65s/it]

    2.62 Seconds to search and locate 50 windows...
    1 cars found


     20%|█▉        | 246/1261 [15:30<1:01:47,  3.65s/it]

    2.6 Seconds to search and locate 54 windows...
    1 cars found


     20%|█▉        | 247/1261 [15:34<1:01:39,  3.65s/it]

    2.63 Seconds to search and locate 53 windows...
    1 cars found


     20%|█▉        | 248/1261 [15:37<1:01:45,  3.66s/it]

    2.68 Seconds to search and locate 55 windows...
    1 cars found


     20%|█▉        | 249/1261 [15:41<1:01:27,  3.64s/it]

    2.6 Seconds to search and locate 67 windows...
    1 cars found


     20%|█▉        | 250/1261 [15:44<1:01:18,  3.64s/it]

    2.56 Seconds to search and locate 62 windows...
    1 cars found


     20%|█▉        | 251/1261 [15:48<1:01:36,  3.66s/it]

    2.7 Seconds to search and locate 69 windows...
    1 cars found


     20%|█▉        | 252/1261 [15:52<1:01:43,  3.67s/it]

    2.67 Seconds to search and locate 71 windows...
    1 cars found


     20%|██        | 253/1261 [15:55<1:01:13,  3.64s/it]

    2.59 Seconds to search and locate 74 windows...
    1 cars found


     20%|██        | 254/1261 [15:59<1:01:28,  3.66s/it]

    2.64 Seconds to search and locate 70 windows...
    1 cars found


     20%|██        | 255/1261 [16:03<1:01:42,  3.68s/it]

    2.7 Seconds to search and locate 79 windows...
    1 cars found


     20%|██        | 256/1261 [16:07<1:01:37,  3.68s/it]

    2.68 Seconds to search and locate 88 windows...
    1 cars found


     20%|██        | 257/1261 [16:10<1:01:13,  3.66s/it]

    2.59 Seconds to search and locate 80 windows...
    1 cars found


     20%|██        | 258/1261 [16:14<1:01:03,  3.65s/it]

    2.61 Seconds to search and locate 70 windows...
    1 cars found


     21%|██        | 259/1261 [16:17<1:01:06,  3.66s/it]

    2.68 Seconds to search and locate 75 windows...
    1 cars found


     21%|██        | 260/1261 [16:21<1:00:50,  3.65s/it]

    2.61 Seconds to search and locate 76 windows...
    1 cars found


     21%|██        | 261/1261 [16:25<1:00:42,  3.64s/it]

    2.58 Seconds to search and locate 90 windows...
    1 cars found


     21%|██        | 262/1261 [16:28<1:01:03,  3.67s/it]

    2.69 Seconds to search and locate 84 windows...
    1 cars found


     21%|██        | 263/1261 [16:32<1:01:18,  3.69s/it]

    2.72 Seconds to search and locate 83 windows...
    1 cars found


     21%|██        | 264/1261 [16:36<1:00:59,  3.67s/it]

    2.62 Seconds to search and locate 84 windows...
    1 cars found


     21%|██        | 265/1261 [16:40<1:01:52,  3.73s/it]

    2.77 Seconds to search and locate 90 windows...
    1 cars found


     21%|██        | 266/1261 [16:43<1:02:06,  3.75s/it]

    2.78 Seconds to search and locate 85 windows...
    1 cars found


     21%|██        | 267/1261 [16:47<1:02:20,  3.76s/it]

    2.79 Seconds to search and locate 95 windows...
    1 cars found


     21%|██▏       | 268/1261 [16:51<1:01:33,  3.72s/it]

    2.6 Seconds to search and locate 77 windows...
    1 cars found


     21%|██▏       | 269/1261 [16:55<1:01:13,  3.70s/it]

    2.6 Seconds to search and locate 83 windows...
    1 cars found


     21%|██▏       | 270/1261 [16:58<1:01:09,  3.70s/it]

    2.71 Seconds to search and locate 73 windows...
    1 cars found


     21%|██▏       | 271/1261 [17:02<1:01:13,  3.71s/it]

    2.73 Seconds to search and locate 65 windows...
    1 cars found


     22%|██▏       | 272/1261 [17:06<1:00:26,  3.67s/it]

    2.58 Seconds to search and locate 62 windows...
    1 cars found


     22%|██▏       | 273/1261 [17:09<1:00:30,  3.67s/it]

    2.63 Seconds to search and locate 64 windows...
    1 cars found


     22%|██▏       | 274/1261 [17:13<1:00:28,  3.68s/it]

    2.69 Seconds to search and locate 76 windows...
    1 cars found


     22%|██▏       | 275/1261 [17:17<1:00:11,  3.66s/it]

    2.65 Seconds to search and locate 73 windows...
    1 cars found


     22%|██▏       | 276/1261 [17:20<1:00:05,  3.66s/it]

    2.61 Seconds to search and locate 74 windows...
    1 cars found


     22%|██▏       | 277/1261 [17:24<1:00:00,  3.66s/it]

    2.64 Seconds to search and locate 71 windows...
    1 cars found


     22%|██▏       | 278/1261 [17:28<1:00:11,  3.67s/it]

    2.72 Seconds to search and locate 79 windows...
    1 cars found


     22%|██▏       | 279/1261 [17:31<1:00:04,  3.67s/it]

    2.66 Seconds to search and locate 86 windows...
    1 cars found


     22%|██▏       | 280/1261 [17:35<59:47,  3.66s/it]  

    2.59 Seconds to search and locate 80 windows...
    1 cars found


     22%|██▏       | 281/1261 [17:39<59:58,  3.67s/it]

    2.69 Seconds to search and locate 76 windows...
    1 cars found


     22%|██▏       | 282/1261 [17:42<1:00:10,  3.69s/it]

    2.71 Seconds to search and locate 75 windows...
    1 cars found


     22%|██▏       | 283/1261 [17:46<59:36,  3.66s/it]  

    2.59 Seconds to search and locate 82 windows...
    1 cars found


     23%|██▎       | 284/1261 [17:50<59:49,  3.67s/it]

    2.64 Seconds to search and locate 85 windows...
    1 cars found


     23%|██▎       | 285/1261 [17:53<59:50,  3.68s/it]

    2.69 Seconds to search and locate 79 windows...
    1 cars found


     23%|██▎       | 286/1261 [17:57<59:43,  3.67s/it]

    2.67 Seconds to search and locate 88 windows...
    1 cars found


     23%|██▎       | 287/1261 [18:01<59:32,  3.67s/it]

    2.63 Seconds to search and locate 83 windows...
    1 cars found


     23%|██▎       | 288/1261 [18:04<59:27,  3.67s/it]

    2.6 Seconds to search and locate 91 windows...
    1 cars found


     23%|██▎       | 289/1261 [18:08<59:39,  3.68s/it]

    2.7 Seconds to search and locate 85 windows...
    1 cars found


     23%|██▎       | 290/1261 [18:12<59:40,  3.69s/it]

    2.68 Seconds to search and locate 88 windows...
    1 cars found


     23%|██▎       | 291/1261 [18:15<59:10,  3.66s/it]

    2.59 Seconds to search and locate 90 windows...
    1 cars found


     23%|██▎       | 292/1261 [18:19<59:42,  3.70s/it]

    2.73 Seconds to search and locate 90 windows...
    1 cars found


     23%|██▎       | 293/1261 [18:23<59:45,  3.70s/it]

    2.71 Seconds to search and locate 91 windows...
    1 cars found


     23%|██▎       | 294/1261 [18:26<59:35,  3.70s/it]

    2.7 Seconds to search and locate 81 windows...
    1 cars found


     23%|██▎       | 295/1261 [18:30<59:26,  3.69s/it]

    2.64 Seconds to search and locate 87 windows...
    1 cars found


     23%|██▎       | 296/1261 [18:34<59:18,  3.69s/it]

    2.66 Seconds to search and locate 80 windows...
    1 cars found


     24%|██▎       | 297/1261 [18:38<59:24,  3.70s/it]

    2.73 Seconds to search and locate 90 windows...
    1 cars found


     24%|██▎       | 298/1261 [18:41<1:00:04,  3.74s/it]

    2.82 Seconds to search and locate 83 windows...
    1 cars found


     24%|██▎       | 299/1261 [18:45<1:00:04,  3.75s/it]

    2.69 Seconds to search and locate 88 windows...
    1 cars found


     24%|██▍       | 300/1261 [18:49<1:00:00,  3.75s/it]

    2.68 Seconds to search and locate 85 windows...
    1 cars found


     24%|██▍       | 301/1261 [18:53<59:33,  3.72s/it]  

    2.67 Seconds to search and locate 61 windows...
    1 cars found


     24%|██▍       | 302/1261 [18:56<59:18,  3.71s/it]

    2.65 Seconds to search and locate 65 windows...
    1 cars found


     24%|██▍       | 303/1261 [19:00<59:10,  3.71s/it]

    2.65 Seconds to search and locate 72 windows...
    1 cars found


     24%|██▍       | 304/1261 [19:04<59:04,  3.70s/it]

    2.67 Seconds to search and locate 79 windows...
    1 cars found


     24%|██▍       | 305/1261 [19:07<59:05,  3.71s/it]

    2.74 Seconds to search and locate 86 windows...
    1 cars found


     24%|██▍       | 306/1261 [19:11<58:39,  3.69s/it]

    2.63 Seconds to search and locate 76 windows...
    1 cars found


     24%|██▍       | 307/1261 [19:15<58:31,  3.68s/it]

    2.58 Seconds to search and locate 76 windows...
    1 cars found


     24%|██▍       | 308/1261 [19:18<58:38,  3.69s/it]

    2.7 Seconds to search and locate 68 windows...
    1 cars found


     25%|██▍       | 309/1261 [19:22<58:37,  3.70s/it]

    2.7 Seconds to search and locate 76 windows...
    1 cars found


     25%|██▍       | 310/1261 [19:26<58:08,  3.67s/it]

    2.58 Seconds to search and locate 69 windows...
    1 cars found


     25%|██▍       | 311/1261 [19:29<58:32,  3.70s/it]

    2.69 Seconds to search and locate 67 windows...
    1 cars found


     25%|██▍       | 312/1261 [19:33<58:32,  3.70s/it]

    2.67 Seconds to search and locate 72 windows...
    1 cars found


     25%|██▍       | 313/1261 [19:37<58:23,  3.70s/it]

    2.67 Seconds to search and locate 77 windows...
    1 cars found


     25%|██▍       | 314/1261 [19:40<58:02,  3.68s/it]

    2.61 Seconds to search and locate 72 windows...
    2 cars found


     25%|██▍       | 315/1261 [19:44<58:00,  3.68s/it]

    2.64 Seconds to search and locate 77 windows...
    1 cars found


     25%|██▌       | 316/1261 [19:48<58:02,  3.68s/it]

    2.68 Seconds to search and locate 70 windows...
    1 cars found


     25%|██▌       | 317/1261 [19:52<57:56,  3.68s/it]

    2.66 Seconds to search and locate 71 windows...
    1 cars found


     25%|██▌       | 318/1261 [19:55<57:35,  3.66s/it]

    2.59 Seconds to search and locate 70 windows...
    1 cars found


     25%|██▌       | 319/1261 [19:59<57:44,  3.68s/it]

    2.67 Seconds to search and locate 75 windows...
    1 cars found


     25%|██▌       | 320/1261 [20:03<57:59,  3.70s/it]

    2.74 Seconds to search and locate 69 windows...
    1 cars found


     25%|██▌       | 321/1261 [20:06<57:48,  3.69s/it]

    2.66 Seconds to search and locate 65 windows...
    1 cars found


     26%|██▌       | 322/1261 [20:10<57:39,  3.68s/it]

    2.62 Seconds to search and locate 70 windows...
    1 cars found


     26%|██▌       | 323/1261 [20:14<57:34,  3.68s/it]

    2.66 Seconds to search and locate 67 windows...
    1 cars found


     26%|██▌       | 324/1261 [20:17<57:38,  3.69s/it]

    2.72 Seconds to search and locate 65 windows...
    1 cars found


     26%|██▌       | 325/1261 [20:21<57:17,  3.67s/it]

    2.61 Seconds to search and locate 64 windows...
    1 cars found


     26%|██▌       | 326/1261 [20:25<57:13,  3.67s/it]

    2.6 Seconds to search and locate 69 windows...
    1 cars found


     26%|██▌       | 327/1261 [20:28<57:30,  3.69s/it]

    2.71 Seconds to search and locate 70 windows...
    1 cars found


     26%|██▌       | 328/1261 [20:32<57:38,  3.71s/it]

    2.74 Seconds to search and locate 75 windows...
    1 cars found


     26%|██▌       | 329/1261 [20:36<57:04,  3.67s/it]

    2.6 Seconds to search and locate 72 windows...
    1 cars found


     26%|██▌       | 330/1261 [20:40<58:17,  3.76s/it]

    2.87 Seconds to search and locate 72 windows...
    1 cars found


     26%|██▌       | 331/1261 [20:44<58:34,  3.78s/it]

    2.8 Seconds to search and locate 51 windows...
    1 cars found


     26%|██▋       | 332/1261 [20:47<58:41,  3.79s/it]

    2.8 Seconds to search and locate 51 windows...
    1 cars found


     26%|██▋       | 333/1261 [20:51<57:55,  3.75s/it]

    2.63 Seconds to search and locate 52 windows...
    1 cars found


     26%|██▋       | 334/1261 [20:55<57:39,  3.73s/it]

    2.6 Seconds to search and locate 55 windows...
    1 cars found


     27%|██▋       | 335/1261 [20:58<57:38,  3.74s/it]

    2.72 Seconds to search and locate 57 windows...
    1 cars found


     27%|██▋       | 336/1261 [21:02<57:33,  3.73s/it]

    2.72 Seconds to search and locate 57 windows...
    1 cars found


     27%|██▋       | 337/1261 [21:06<56:44,  3.68s/it]

    2.57 Seconds to search and locate 58 windows...
    1 cars found


     27%|██▋       | 338/1261 [21:09<56:51,  3.70s/it]

    2.63 Seconds to search and locate 55 windows...
    1 cars found


     27%|██▋       | 339/1261 [21:13<56:44,  3.69s/it]

    2.68 Seconds to search and locate 55 windows...
    1 cars found


     27%|██▋       | 340/1261 [21:17<56:39,  3.69s/it]

    2.68 Seconds to search and locate 55 windows...
    1 cars found


     27%|██▋       | 341/1261 [21:20<56:18,  3.67s/it]

    2.62 Seconds to search and locate 55 windows...
    1 cars found


     27%|██▋       | 342/1261 [21:24<56:23,  3.68s/it]

    2.63 Seconds to search and locate 59 windows...
    1 cars found


     27%|██▋       | 343/1261 [21:28<56:36,  3.70s/it]

    2.71 Seconds to search and locate 64 windows...
    1 cars found


     27%|██▋       | 344/1261 [21:32<56:40,  3.71s/it]

    2.7 Seconds to search and locate 73 windows...
    1 cars found


     27%|██▋       | 345/1261 [21:35<56:16,  3.69s/it]

    2.6 Seconds to search and locate 79 windows...
    1 cars found


     27%|██▋       | 346/1261 [21:39<56:12,  3.69s/it]

    2.65 Seconds to search and locate 67 windows...
    1 cars found


     28%|██▊       | 347/1261 [21:43<56:13,  3.69s/it]

    2.69 Seconds to search and locate 54 windows...
    1 cars found


     28%|██▊       | 348/1261 [21:46<55:44,  3.66s/it]

    2.59 Seconds to search and locate 52 windows...
    1 cars found


     28%|██▊       | 349/1261 [21:50<55:45,  3.67s/it]

    2.61 Seconds to search and locate 67 windows...
    1 cars found


     28%|██▊       | 350/1261 [21:54<55:45,  3.67s/it]

    2.67 Seconds to search and locate 60 windows...
    1 cars found


     28%|██▊       | 351/1261 [21:57<55:56,  3.69s/it]

    2.71 Seconds to search and locate 54 windows...
    1 cars found


     28%|██▊       | 352/1261 [22:01<56:01,  3.70s/it]

    2.7 Seconds to search and locate 58 windows...
    1 cars found


     28%|██▊       | 353/1261 [22:05<56:05,  3.71s/it]

    2.63 Seconds to search and locate 54 windows...
    1 cars found


     28%|██▊       | 354/1261 [22:09<56:15,  3.72s/it]

    2.74 Seconds to search and locate 55 windows...
    1 cars found


     28%|██▊       | 355/1261 [22:12<56:04,  3.71s/it]

    2.7 Seconds to search and locate 52 windows...
    1 cars found


     28%|██▊       | 356/1261 [22:16<55:26,  3.68s/it]

    2.6 Seconds to search and locate 50 windows...
    1 cars found


     28%|██▊       | 357/1261 [22:20<55:32,  3.69s/it]

    2.63 Seconds to search and locate 49 windows...
    1 cars found


     28%|██▊       | 358/1261 [22:23<55:32,  3.69s/it]

    2.68 Seconds to search and locate 44 windows...
    1 cars found


     28%|██▊       | 359/1261 [22:27<55:28,  3.69s/it]

    2.68 Seconds to search and locate 39 windows...
    1 cars found


     29%|██▊       | 360/1261 [22:31<55:18,  3.68s/it]

    2.64 Seconds to search and locate 47 windows...
    1 cars found


     29%|██▊       | 361/1261 [22:34<55:12,  3.68s/it]

    2.6 Seconds to search and locate 49 windows...
    1 cars found


     29%|██▊       | 362/1261 [22:38<55:31,  3.71s/it]

    2.77 Seconds to search and locate 52 windows...
    1 cars found


     29%|██▉       | 363/1261 [22:42<55:52,  3.73s/it]

    2.75 Seconds to search and locate 55 windows...
    1 cars found


     29%|██▉       | 364/1261 [22:46<56:15,  3.76s/it]

    2.77 Seconds to search and locate 54 windows...
    1 cars found


     29%|██▉       | 365/1261 [22:49<56:08,  3.76s/it]

    2.65 Seconds to search and locate 59 windows...
    1 cars found


     29%|██▉       | 366/1261 [22:53<55:42,  3.73s/it]

    2.67 Seconds to search and locate 51 windows...
    1 cars found


     29%|██▉       | 367/1261 [22:57<55:21,  3.72s/it]

    2.66 Seconds to search and locate 51 windows...
    1 cars found


     29%|██▉       | 368/1261 [23:00<55:08,  3.71s/it]

    2.65 Seconds to search and locate 51 windows...
    1 cars found


     29%|██▉       | 369/1261 [23:04<55:09,  3.71s/it]

    2.67 Seconds to search and locate 54 windows...
    1 cars found


     29%|██▉       | 370/1261 [23:08<55:10,  3.72s/it]

    2.72 Seconds to search and locate 50 windows...
    1 cars found


     29%|██▉       | 371/1261 [23:12<54:59,  3.71s/it]

    2.66 Seconds to search and locate 54 windows...
    1 cars found


     30%|██▉       | 372/1261 [23:15<54:29,  3.68s/it]

    2.6 Seconds to search and locate 58 windows...
    1 cars found


     30%|██▉       | 373/1261 [23:19<54:24,  3.68s/it]

    2.67 Seconds to search and locate 56 windows...
    1 cars found


     30%|██▉       | 374/1261 [23:23<54:36,  3.69s/it]

    2.72 Seconds to search and locate 55 windows...
    1 cars found


     30%|██▉       | 375/1261 [23:26<54:10,  3.67s/it]

    2.64 Seconds to search and locate 55 windows...
    1 cars found


     30%|██▉       | 376/1261 [23:30<54:14,  3.68s/it]

    2.64 Seconds to search and locate 57 windows...
    1 cars found


     30%|██▉       | 377/1261 [23:34<54:28,  3.70s/it]

    2.72 Seconds to search and locate 56 windows...
    1 cars found


     30%|██▉       | 378/1261 [23:37<54:36,  3.71s/it]

    2.73 Seconds to search and locate 55 windows...
    1 cars found


     30%|███       | 379/1261 [23:41<54:19,  3.70s/it]

    2.65 Seconds to search and locate 53 windows...
    1 cars found


     30%|███       | 380/1261 [23:45<54:15,  3.70s/it]

    2.58 Seconds to search and locate 54 windows...
    1 cars found


     30%|███       | 381/1261 [23:48<54:27,  3.71s/it]

    2.75 Seconds to search and locate 51 windows...
    1 cars found


     30%|███       | 382/1261 [23:52<54:34,  3.73s/it]

    2.72 Seconds to search and locate 51 windows...
    1 cars found


     30%|███       | 383/1261 [23:56<53:54,  3.68s/it]

    2.58 Seconds to search and locate 46 windows...
    1 cars found


     30%|███       | 384/1261 [24:00<54:04,  3.70s/it]

    2.67 Seconds to search and locate 47 windows...
    1 cars found


     31%|███       | 385/1261 [24:03<54:18,  3.72s/it]

    2.75 Seconds to search and locate 53 windows...
    1 cars found


     31%|███       | 386/1261 [24:07<54:07,  3.71s/it]

    2.68 Seconds to search and locate 59 windows...
    1 cars found


     31%|███       | 387/1261 [24:11<53:51,  3.70s/it]

    2.62 Seconds to search and locate 63 windows...
    1 cars found


     31%|███       | 388/1261 [24:14<53:42,  3.69s/it]

    2.61 Seconds to search and locate 54 windows...
    1 cars found


     31%|███       | 389/1261 [24:18<53:38,  3.69s/it]

    2.7 Seconds to search and locate 56 windows...
    1 cars found


     31%|███       | 390/1261 [24:22<53:41,  3.70s/it]

    2.68 Seconds to search and locate 55 windows...
    1 cars found


     31%|███       | 391/1261 [24:25<53:08,  3.66s/it]

    2.57 Seconds to search and locate 48 windows...
    1 cars found


     31%|███       | 392/1261 [24:29<53:12,  3.67s/it]

    2.65 Seconds to search and locate 48 windows...
    1 cars found


     31%|███       | 393/1261 [24:33<53:30,  3.70s/it]

    2.71 Seconds to search and locate 49 windows...
    1 cars found


     31%|███       | 394/1261 [24:36<53:10,  3.68s/it]

    2.63 Seconds to search and locate 53 windows...
    1 cars found


     31%|███▏      | 395/1261 [24:40<53:50,  3.73s/it]

    2.78 Seconds to search and locate 49 windows...
    1 cars found


     31%|███▏      | 396/1261 [24:44<53:53,  3.74s/it]

    2.69 Seconds to search and locate 51 windows...
    1 cars found


     31%|███▏      | 397/1261 [24:48<53:58,  3.75s/it]

    2.74 Seconds to search and locate 47 windows...
    1 cars found


     32%|███▏      | 398/1261 [24:52<53:44,  3.74s/it]

    2.7 Seconds to search and locate 49 windows...
    1 cars found


     32%|███▏      | 399/1261 [24:55<53:11,  3.70s/it]

    2.6 Seconds to search and locate 43 windows...
    1 cars found


     32%|███▏      | 400/1261 [24:59<53:08,  3.70s/it]

    2.69 Seconds to search and locate 45 windows...
    1 cars found


     32%|███▏      | 401/1261 [25:03<53:03,  3.70s/it]

    2.7 Seconds to search and locate 44 windows...
    1 cars found


     32%|███▏      | 402/1261 [25:06<52:36,  3.67s/it]

    2.61 Seconds to search and locate 41 windows...
    1 cars found


     32%|███▏      | 403/1261 [25:10<52:36,  3.68s/it]

    2.63 Seconds to search and locate 45 windows...
    1 cars found


     32%|███▏      | 404/1261 [25:14<52:41,  3.69s/it]

    2.7 Seconds to search and locate 44 windows...
    1 cars found


     32%|███▏      | 405/1261 [25:17<52:41,  3.69s/it]

    2.72 Seconds to search and locate 44 windows...
    1 cars found


     32%|███▏      | 406/1261 [25:21<52:16,  3.67s/it]

    2.61 Seconds to search and locate 40 windows...
    1 cars found


     32%|███▏      | 407/1261 [25:25<52:06,  3.66s/it]

    2.58 Seconds to search and locate 39 windows...
    1 cars found


     32%|███▏      | 408/1261 [25:28<52:16,  3.68s/it]

    2.7 Seconds to search and locate 39 windows...
    1 cars found


     32%|███▏      | 409/1261 [25:32<52:31,  3.70s/it]

    2.72 Seconds to search and locate 40 windows...
    1 cars found


     33%|███▎      | 410/1261 [25:36<52:02,  3.67s/it]

    2.6 Seconds to search and locate 45 windows...
    1 cars found


     33%|███▎      | 411/1261 [25:39<52:12,  3.69s/it]

    2.66 Seconds to search and locate 46 windows...
    1 cars found


     33%|███▎      | 412/1261 [25:43<52:10,  3.69s/it]

    2.69 Seconds to search and locate 43 windows...
    1 cars found


     33%|███▎      | 413/1261 [25:47<51:49,  3.67s/it]

    2.62 Seconds to search and locate 41 windows...
    1 cars found


     33%|███▎      | 414/1261 [25:50<51:39,  3.66s/it]

    2.6 Seconds to search and locate 42 windows...
    1 cars found


     33%|███▎      | 415/1261 [25:54<51:45,  3.67s/it]

    2.66 Seconds to search and locate 42 windows...
    1 cars found


     33%|███▎      | 416/1261 [25:58<51:53,  3.68s/it]

    2.7 Seconds to search and locate 32 windows...
    1 cars found


     33%|███▎      | 417/1261 [26:01<51:45,  3.68s/it]

    2.63 Seconds to search and locate 41 windows...
    1 cars found


     33%|███▎      | 418/1261 [26:05<51:31,  3.67s/it]

    2.57 Seconds to search and locate 41 windows...
    1 cars found


     33%|███▎      | 419/1261 [26:09<51:39,  3.68s/it]

    2.7 Seconds to search and locate 42 windows...
    1 cars found


     33%|███▎      | 420/1261 [26:12<51:48,  3.70s/it]

    2.72 Seconds to search and locate 37 windows...
    1 cars found


     33%|███▎      | 421/1261 [26:16<51:16,  3.66s/it]

    2.58 Seconds to search and locate 33 windows...
    1 cars found


     33%|███▎      | 422/1261 [26:20<51:22,  3.67s/it]

    2.61 Seconds to search and locate 38 windows...
    1 cars found


     34%|███▎      | 423/1261 [26:23<51:29,  3.69s/it]

    2.71 Seconds to search and locate 36 windows...
    1 cars found


     34%|███▎      | 424/1261 [26:27<51:32,  3.70s/it]

    2.7 Seconds to search and locate 41 windows...
    1 cars found


     34%|███▎      | 425/1261 [26:31<51:20,  3.68s/it]

    2.62 Seconds to search and locate 39 windows...
    1 cars found


     34%|███▍      | 426/1261 [26:35<51:29,  3.70s/it]

    2.66 Seconds to search and locate 38 windows...
    1 cars found


     34%|███▍      | 427/1261 [26:38<51:48,  3.73s/it]

    2.8 Seconds to search and locate 39 windows...
    1 cars found


     34%|███▍      | 428/1261 [26:42<52:13,  3.76s/it]

    2.78 Seconds to search and locate 36 windows...
    1 cars found


     34%|███▍      | 429/1261 [26:46<52:05,  3.76s/it]

    2.71 Seconds to search and locate 39 windows...
    1 cars found


     34%|███▍      | 430/1261 [26:50<52:02,  3.76s/it]

    2.64 Seconds to search and locate 33 windows...
    1 cars found


     34%|███▍      | 431/1261 [26:53<51:37,  3.73s/it]

    2.67 Seconds to search and locate 34 windows...
    1 cars found


     34%|███▍      | 432/1261 [26:57<51:23,  3.72s/it]

    2.69 Seconds to search and locate 37 windows...
    1 cars found


     34%|███▍      | 433/1261 [27:01<51:01,  3.70s/it]

    2.65 Seconds to search and locate 35 windows...
    1 cars found


     34%|███▍      | 434/1261 [27:04<50:57,  3.70s/it]

    2.63 Seconds to search and locate 42 windows...
    1 cars found


     34%|███▍      | 435/1261 [27:08<50:51,  3.69s/it]

    2.67 Seconds to search and locate 43 windows...
    1 cars found


     35%|███▍      | 436/1261 [27:12<50:52,  3.70s/it]

    2.69 Seconds to search and locate 41 windows...
    1 cars found


     35%|███▍      | 437/1261 [27:15<50:28,  3.68s/it]

    2.6 Seconds to search and locate 39 windows...
    1 cars found


     35%|███▍      | 438/1261 [27:19<50:33,  3.69s/it]

    2.68 Seconds to search and locate 36 windows...
    1 cars found


     35%|███▍      | 439/1261 [27:23<50:31,  3.69s/it]

    2.67 Seconds to search and locate 31 windows...
    1 cars found


     35%|███▍      | 440/1261 [27:26<50:15,  3.67s/it]

    2.63 Seconds to search and locate 37 windows...
    1 cars found


     35%|███▍      | 441/1261 [27:30<50:20,  3.68s/it]

    2.64 Seconds to search and locate 38 windows...
    1 cars found


     35%|███▌      | 442/1261 [27:34<50:23,  3.69s/it]

    2.69 Seconds to search and locate 35 windows...
    1 cars found


     35%|███▌      | 443/1261 [27:38<50:19,  3.69s/it]

    2.68 Seconds to search and locate 37 windows...
    1 cars found


     35%|███▌      | 444/1261 [27:41<50:09,  3.68s/it]

    2.62 Seconds to search and locate 34 windows...
    1 cars found


     35%|███▌      | 445/1261 [27:45<49:59,  3.68s/it]

    2.6 Seconds to search and locate 34 windows...
    1 cars found


     35%|███▌      | 446/1261 [27:49<50:03,  3.69s/it]

    2.68 Seconds to search and locate 37 windows...
    1 cars found


     35%|███▌      | 447/1261 [27:52<50:12,  3.70s/it]

    2.71 Seconds to search and locate 37 windows...
    1 cars found


     36%|███▌      | 448/1261 [27:56<49:51,  3.68s/it]

    2.62 Seconds to search and locate 35 windows...
    1 cars found


     36%|███▌      | 449/1261 [28:00<49:53,  3.69s/it]

    2.62 Seconds to search and locate 34 windows...
    1 cars found


     36%|███▌      | 450/1261 [28:03<49:59,  3.70s/it]

    2.7 Seconds to search and locate 39 windows...
    1 cars found


     36%|███▌      | 451/1261 [28:07<49:55,  3.70s/it]

    2.68 Seconds to search and locate 34 windows...
    1 cars found


     36%|███▌      | 452/1261 [28:11<49:38,  3.68s/it]

    2.62 Seconds to search and locate 32 windows...
    1 cars found


     36%|███▌      | 453/1261 [28:14<49:28,  3.67s/it]

    2.6 Seconds to search and locate 31 windows...
    1 cars found


     36%|███▌      | 454/1261 [28:18<49:25,  3.68s/it]

    2.67 Seconds to search and locate 34 windows...
    1 cars found


     36%|███▌      | 455/1261 [28:22<49:29,  3.68s/it]

    2.68 Seconds to search and locate 36 windows...
    1 cars found


     36%|███▌      | 456/1261 [28:25<49:07,  3.66s/it]

    2.57 Seconds to search and locate 32 windows...
    1 cars found


     36%|███▌      | 457/1261 [28:29<49:04,  3.66s/it]

    2.64 Seconds to search and locate 35 windows...
    1 cars found


     36%|███▋      | 458/1261 [28:33<49:16,  3.68s/it]

    2.7 Seconds to search and locate 38 windows...
    1 cars found


     36%|███▋      | 459/1261 [28:36<48:53,  3.66s/it]

    2.57 Seconds to search and locate 34 windows...
    1 cars found


     36%|███▋      | 460/1261 [28:40<49:41,  3.72s/it]

    2.8 Seconds to search and locate 31 windows...
    1 cars found


     37%|███▋      | 461/1261 [28:44<49:58,  3.75s/it]

    2.74 Seconds to search and locate 27 windows...
    1 cars found


     37%|███▋      | 462/1261 [28:48<50:14,  3.77s/it]

    2.72 Seconds to search and locate 34 windows...
    1 cars found


     37%|███▋      | 463/1261 [28:52<49:48,  3.75s/it]

    2.62 Seconds to search and locate 33 windows...
    1 cars found


     37%|███▋      | 464/1261 [28:55<49:15,  3.71s/it]

    2.58 Seconds to search and locate 33 windows...
    1 cars found


     37%|███▋      | 465/1261 [28:59<49:06,  3.70s/it]

    2.66 Seconds to search and locate 33 windows...
    1 cars found


     37%|███▋      | 466/1261 [29:03<49:33,  3.74s/it]

    2.74 Seconds to search and locate 34 windows...
    1 cars found


     37%|███▋      | 467/1261 [29:06<49:36,  3.75s/it]

    2.61 Seconds to search and locate 31 windows...
    1 cars found


     37%|███▋      | 468/1261 [29:10<49:13,  3.72s/it]

    2.64 Seconds to search and locate 35 windows...
    1 cars found


     37%|███▋      | 469/1261 [29:14<48:52,  3.70s/it]

    2.66 Seconds to search and locate 34 windows...
    1 cars found


     37%|███▋      | 470/1261 [29:17<48:36,  3.69s/it]

    2.67 Seconds to search and locate 36 windows...
    1 cars found


     37%|███▋      | 471/1261 [29:21<48:13,  3.66s/it]

    2.62 Seconds to search and locate 36 windows...
    1 cars found


     37%|███▋      | 472/1261 [29:25<48:02,  3.65s/it]

    2.56 Seconds to search and locate 42 windows...
    1 cars found


     38%|███▊      | 473/1261 [29:28<48:00,  3.66s/it]

    2.66 Seconds to search and locate 37 windows...
    1 cars found


     38%|███▊      | 474/1261 [29:32<48:08,  3.67s/it]

    2.7 Seconds to search and locate 40 windows...
    1 cars found


     38%|███▊      | 475/1261 [29:36<47:48,  3.65s/it]

    2.59 Seconds to search and locate 37 windows...
    1 cars found


     38%|███▊      | 476/1261 [29:39<47:52,  3.66s/it]

    2.63 Seconds to search and locate 39 windows...
    1 cars found


     38%|███▊      | 477/1261 [29:43<48:10,  3.69s/it]

    2.75 Seconds to search and locate 34 windows...
    1 cars found


     38%|███▊      | 478/1261 [29:47<47:55,  3.67s/it]

    2.64 Seconds to search and locate 39 windows...
    1 cars found


     38%|███▊      | 479/1261 [29:50<47:40,  3.66s/it]

    2.61 Seconds to search and locate 39 windows...
    1 cars found


     38%|███▊      | 480/1261 [29:54<47:32,  3.65s/it]

    2.64 Seconds to search and locate 37 windows...
    1 cars found


     38%|███▊      | 481/1261 [29:58<47:34,  3.66s/it]

    2.68 Seconds to search and locate 27 windows...
    1 cars found


     38%|███▊      | 482/1261 [30:01<47:23,  3.65s/it]

    2.62 Seconds to search and locate 36 windows...
    1 cars found


     38%|███▊      | 483/1261 [30:05<47:17,  3.65s/it]

    2.59 Seconds to search and locate 33 windows...
    1 cars found


     38%|███▊      | 484/1261 [30:09<47:30,  3.67s/it]

    2.71 Seconds to search and locate 33 windows...
    1 cars found


     38%|███▊      | 485/1261 [30:12<47:31,  3.67s/it]

    2.68 Seconds to search and locate 31 windows...
    1 cars found


     39%|███▊      | 486/1261 [30:16<47:06,  3.65s/it]

    2.58 Seconds to search and locate 33 windows...
    1 cars found


     39%|███▊      | 487/1261 [30:20<47:08,  3.65s/it]

    2.62 Seconds to search and locate 42 windows...
    2 cars found


     39%|███▊      | 488/1261 [30:23<47:07,  3.66s/it]

    2.67 Seconds to search and locate 33 windows...
    1 cars found


     39%|███▉      | 489/1261 [30:27<47:02,  3.66s/it]

    2.65 Seconds to search and locate 40 windows...
    2 cars found


     39%|███▉      | 490/1261 [30:31<46:49,  3.64s/it]

    2.6 Seconds to search and locate 43 windows...
    2 cars found


     39%|███▉      | 491/1261 [30:34<46:55,  3.66s/it]

    2.68 Seconds to search and locate 40 windows...
    2 cars found


     39%|███▉      | 492/1261 [30:38<47:07,  3.68s/it]

    2.73 Seconds to search and locate 32 windows...
    1 cars found


     39%|███▉      | 493/1261 [30:42<47:03,  3.68s/it]

    2.67 Seconds to search and locate 33 windows...
    1 cars found


     39%|███▉      | 494/1261 [30:45<46:43,  3.66s/it]

    2.57 Seconds to search and locate 34 windows...
    1 cars found


     39%|███▉      | 495/1261 [30:49<46:42,  3.66s/it]

    2.67 Seconds to search and locate 32 windows...
    1 cars found


     39%|███▉      | 496/1261 [30:53<46:44,  3.67s/it]

    2.69 Seconds to search and locate 39 windows...
    1 cars found


     39%|███▉      | 497/1261 [30:56<46:21,  3.64s/it]

    2.59 Seconds to search and locate 33 windows...
    1 cars found


     39%|███▉      | 498/1261 [31:00<46:23,  3.65s/it]

    2.61 Seconds to search and locate 37 windows...
    1 cars found


     40%|███▉      | 499/1261 [31:04<46:31,  3.66s/it]

    2.69 Seconds to search and locate 34 windows...
    1 cars found


     40%|███▉      | 500/1261 [31:07<46:37,  3.68s/it]

    2.69 Seconds to search and locate 33 windows...
    1 cars found


     40%|███▉      | 501/1261 [31:11<46:25,  3.67s/it]

    2.64 Seconds to search and locate 29 windows...
    1 cars found


     40%|███▉      | 502/1261 [31:15<46:19,  3.66s/it]

    2.61 Seconds to search and locate 30 windows...
    1 cars found


     40%|███▉      | 503/1261 [31:18<46:20,  3.67s/it]

    2.69 Seconds to search and locate 29 windows...
    1 cars found


     40%|███▉      | 504/1261 [31:22<46:21,  3.67s/it]

    2.66 Seconds to search and locate 30 windows...
    1 cars found


     40%|████      | 505/1261 [31:26<46:06,  3.66s/it]

    2.59 Seconds to search and locate 25 windows...
    1 cars found


     40%|████      | 506/1261 [31:29<46:11,  3.67s/it]

    2.66 Seconds to search and locate 27 windows...
    1 cars found


     40%|████      | 507/1261 [31:33<46:12,  3.68s/it]

    2.69 Seconds to search and locate 23 windows...
    1 cars found


     40%|████      | 508/1261 [31:37<45:56,  3.66s/it]

    2.61 Seconds to search and locate 26 windows...
    1 cars found


     40%|████      | 509/1261 [31:40<45:48,  3.66s/it]

    2.61 Seconds to search and locate 27 windows...
    1 cars found


     40%|████      | 510/1261 [31:44<45:49,  3.66s/it]

    2.66 Seconds to search and locate 27 windows...
    1 cars found


     41%|████      | 511/1261 [31:47<45:41,  3.65s/it]

    2.65 Seconds to search and locate 25 windows...
    1 cars found


     41%|████      | 512/1261 [31:51<45:28,  3.64s/it]

    2.62 Seconds to search and locate 24 windows...
    1 cars found


     41%|████      | 513/1261 [31:55<45:23,  3.64s/it]

    2.58 Seconds to search and locate 26 windows...
    1 cars found


     41%|████      | 514/1261 [31:58<45:22,  3.64s/it]

    2.65 Seconds to search and locate 27 windows...
    1 cars found


     41%|████      | 515/1261 [32:02<45:30,  3.66s/it]

    2.69 Seconds to search and locate 26 windows...
    1 cars found


     41%|████      | 516/1261 [32:06<45:10,  3.64s/it]

    2.54 Seconds to search and locate 24 windows...
    1 cars found


     41%|████      | 517/1261 [32:09<45:06,  3.64s/it]

    2.6 Seconds to search and locate 20 windows...
    1 cars found


     41%|████      | 518/1261 [32:13<45:09,  3.65s/it]

    2.67 Seconds to search and locate 23 windows...
    1 cars found


     41%|████      | 519/1261 [32:17<44:44,  3.62s/it]

    2.57 Seconds to search and locate 23 windows...
    1 cars found


     41%|████      | 520/1261 [32:20<44:44,  3.62s/it]

    2.6 Seconds to search and locate 26 windows...
    1 cars found


     41%|████▏     | 521/1261 [32:24<44:54,  3.64s/it]

    2.66 Seconds to search and locate 25 windows...
    1 cars found


     41%|████▏     | 522/1261 [32:28<45:00,  3.65s/it]

    2.68 Seconds to search and locate 23 windows...
    1 cars found


     41%|████▏     | 523/1261 [32:31<44:47,  3.64s/it]

    2.61 Seconds to search and locate 27 windows...
    1 cars found


     42%|████▏     | 524/1261 [32:35<44:55,  3.66s/it]

    2.62 Seconds to search and locate 26 windows...
    1 cars found


     42%|████▏     | 525/1261 [32:39<45:01,  3.67s/it]

    2.7 Seconds to search and locate 24 windows...
    1 cars found


     42%|████▏     | 526/1261 [32:42<45:07,  3.68s/it]

    2.7 Seconds to search and locate 27 windows...
    1 cars found


     42%|████▏     | 527/1261 [32:46<44:45,  3.66s/it]

    2.58 Seconds to search and locate 25 windows...
    1 cars found


     42%|████▏     | 528/1261 [32:50<44:42,  3.66s/it]

    2.6 Seconds to search and locate 27 windows...
    1 cars found


     42%|████▏     | 529/1261 [32:53<44:51,  3.68s/it]

    2.69 Seconds to search and locate 28 windows...
    1 cars found


     42%|████▏     | 530/1261 [32:57<44:45,  3.67s/it]

    2.65 Seconds to search and locate 29 windows...
    1 cars found


     42%|████▏     | 531/1261 [33:01<44:40,  3.67s/it]

    2.64 Seconds to search and locate 28 windows...
    1 cars found


     42%|████▏     | 532/1261 [33:04<44:48,  3.69s/it]

    2.68 Seconds to search and locate 27 windows...
    1 cars found


     42%|████▏     | 533/1261 [33:08<44:47,  3.69s/it]

    2.69 Seconds to search and locate 26 windows...
    1 cars found


     42%|████▏     | 534/1261 [33:12<44:37,  3.68s/it]

    2.65 Seconds to search and locate 27 windows...
    1 cars found


     42%|████▏     | 535/1261 [33:15<44:30,  3.68s/it]

    2.58 Seconds to search and locate 27 windows...
    1 cars found


     43%|████▎     | 536/1261 [33:19<44:29,  3.68s/it]

    2.68 Seconds to search and locate 25 windows...
    1 cars found


     43%|████▎     | 537/1261 [33:23<44:33,  3.69s/it]

    2.7 Seconds to search and locate 31 windows...
    1 cars found


     43%|████▎     | 538/1261 [33:26<44:08,  3.66s/it]

    2.58 Seconds to search and locate 33 windows...
    1 cars found


     43%|████▎     | 539/1261 [33:30<44:14,  3.68s/it]

    2.63 Seconds to search and locate 33 windows...
    1 cars found


     43%|████▎     | 540/1261 [33:34<44:26,  3.70s/it]

    2.72 Seconds to search and locate 37 windows...
    1 cars found


     43%|████▎     | 541/1261 [33:37<44:16,  3.69s/it]

    2.64 Seconds to search and locate 27 windows...
    1 cars found


     43%|████▎     | 542/1261 [33:41<44:04,  3.68s/it]

    2.64 Seconds to search and locate 26 windows...
    1 cars found


     43%|████▎     | 543/1261 [33:45<44:00,  3.68s/it]

    2.56 Seconds to search and locate 31 windows...
    1 cars found


     43%|████▎     | 544/1261 [33:48<43:58,  3.68s/it]

    2.65 Seconds to search and locate 31 windows...
    1 cars found


     43%|████▎     | 545/1261 [33:52<43:56,  3.68s/it]

    2.67 Seconds to search and locate 26 windows...
    1 cars found


     43%|████▎     | 546/1261 [33:56<43:33,  3.66s/it]

    2.57 Seconds to search and locate 28 windows...
    1 cars found


     43%|████▎     | 547/1261 [33:59<43:46,  3.68s/it]

    2.68 Seconds to search and locate 24 windows...
    1 cars found


     43%|████▎     | 548/1261 [34:03<43:50,  3.69s/it]

    2.7 Seconds to search and locate 29 windows...
    1 cars found


     44%|████▎     | 549/1261 [34:07<43:39,  3.68s/it]

    2.62 Seconds to search and locate 25 windows...
    1 cars found


     44%|████▎     | 550/1261 [34:10<43:22,  3.66s/it]

    2.59 Seconds to search and locate 21 windows...
    1 cars found


     44%|████▎     | 551/1261 [34:14<43:32,  3.68s/it]

    2.7 Seconds to search and locate 21 windows...
    1 cars found


     44%|████▍     | 552/1261 [34:18<43:26,  3.68s/it]

    2.66 Seconds to search and locate 25 windows...
    1 cars found


     44%|████▍     | 553/1261 [34:21<43:11,  3.66s/it]

    2.6 Seconds to search and locate 28 windows...
    1 cars found


     44%|████▍     | 554/1261 [34:25<43:02,  3.65s/it]

    2.57 Seconds to search and locate 26 windows...
    1 cars found


     44%|████▍     | 555/1261 [34:29<43:17,  3.68s/it]

    2.73 Seconds to search and locate 27 windows...
    1 cars found


     44%|████▍     | 556/1261 [34:33<43:36,  3.71s/it]

    2.77 Seconds to search and locate 26 windows...
    1 cars found


     44%|████▍     | 557/1261 [34:36<43:12,  3.68s/it]

    2.58 Seconds to search and locate 26 windows...
    1 cars found


     44%|████▍     | 558/1261 [34:40<43:31,  3.71s/it]

    2.73 Seconds to search and locate 29 windows...
    1 cars found


     44%|████▍     | 559/1261 [34:44<43:39,  3.73s/it]

    2.74 Seconds to search and locate 28 windows...
    1 cars found


     44%|████▍     | 560/1261 [34:48<43:33,  3.73s/it]

    2.71 Seconds to search and locate 27 windows...
    1 cars found


     44%|████▍     | 561/1261 [34:51<43:29,  3.73s/it]

    2.66 Seconds to search and locate 27 windows...
    1 cars found


     45%|████▍     | 562/1261 [34:55<43:18,  3.72s/it]

    2.62 Seconds to search and locate 24 windows...
    1 cars found


     45%|████▍     | 563/1261 [34:59<43:21,  3.73s/it]

    2.73 Seconds to search and locate 25 windows...
    1 cars found


     45%|████▍     | 564/1261 [35:02<43:21,  3.73s/it]

    2.72 Seconds to search and locate 20 windows...
    1 cars found


     45%|████▍     | 565/1261 [35:06<43:06,  3.72s/it]

    2.65 Seconds to search and locate 16 windows...
    1 cars found


     45%|████▍     | 566/1261 [35:10<43:03,  3.72s/it]

    2.65 Seconds to search and locate 27 windows...
    1 cars found


     45%|████▍     | 567/1261 [35:14<42:59,  3.72s/it]

    2.71 Seconds to search and locate 25 windows...
    1 cars found


     45%|████▌     | 568/1261 [35:17<42:51,  3.71s/it]

    2.7 Seconds to search and locate 21 windows...
    1 cars found


     45%|████▌     | 569/1261 [35:21<42:38,  3.70s/it]

    2.66 Seconds to search and locate 19 windows...
    1 cars found


     45%|████▌     | 570/1261 [35:25<42:41,  3.71s/it]

    2.66 Seconds to search and locate 18 windows...
    1 cars found


     45%|████▌     | 571/1261 [35:28<42:25,  3.69s/it]

    2.65 Seconds to search and locate 18 windows...
    1 cars found


     45%|████▌     | 572/1261 [35:32<42:18,  3.68s/it]

    2.65 Seconds to search and locate 17 windows...
    1 cars found


     45%|████▌     | 573/1261 [35:36<42:00,  3.66s/it]

    2.59 Seconds to search and locate 17 windows...
    1 cars found


     46%|████▌     | 574/1261 [35:39<41:53,  3.66s/it]

    2.63 Seconds to search and locate 14 windows...
    1 cars found


     46%|████▌     | 575/1261 [35:43<41:54,  3.67s/it]

    2.65 Seconds to search and locate 16 windows...
    1 cars found


     46%|████▌     | 576/1261 [35:46<41:30,  3.64s/it]

    2.56 Seconds to search and locate 18 windows...
    1 cars found


     46%|████▌     | 577/1261 [35:50<41:29,  3.64s/it]

    2.59 Seconds to search and locate 15 windows...
    1 cars found


     46%|████▌     | 578/1261 [35:54<41:31,  3.65s/it]

    2.66 Seconds to search and locate 18 windows...
    1 cars found


     46%|████▌     | 579/1261 [35:57<41:32,  3.65s/it]

    2.68 Seconds to search and locate 21 windows...
    1 cars found


     46%|████▌     | 580/1261 [36:01<41:26,  3.65s/it]

    2.65 Seconds to search and locate 19 windows...
    1 cars found


     46%|████▌     | 581/1261 [36:05<41:29,  3.66s/it]

    2.62 Seconds to search and locate 18 windows...
    1 cars found


     46%|████▌     | 582/1261 [36:08<41:28,  3.66s/it]

    2.68 Seconds to search and locate 19 windows...
    1 cars found


     46%|████▌     | 583/1261 [36:12<41:38,  3.68s/it]

    2.7 Seconds to search and locate 20 windows...
    1 cars found


     46%|████▋     | 584/1261 [36:16<41:08,  3.65s/it]

    2.55 Seconds to search and locate 24 windows...
    1 cars found


     46%|████▋     | 585/1261 [36:19<41:06,  3.65s/it]

    2.62 Seconds to search and locate 23 windows...
    1 cars found


     46%|████▋     | 586/1261 [36:23<41:18,  3.67s/it]

    2.7 Seconds to search and locate 24 windows...
    1 cars found


     47%|████▋     | 587/1261 [36:27<41:08,  3.66s/it]

    2.65 Seconds to search and locate 23 windows...
    1 cars found


     47%|████▋     | 588/1261 [36:30<40:57,  3.65s/it]

    2.58 Seconds to search and locate 22 windows...
    1 cars found


     47%|████▋     | 589/1261 [36:34<40:59,  3.66s/it]

    2.69 Seconds to search and locate 19 windows...
    1 cars found


     47%|████▋     | 590/1261 [36:38<41:04,  3.67s/it]

    2.7 Seconds to search and locate 15 windows...
    1 cars found


     47%|████▋     | 591/1261 [36:41<40:56,  3.67s/it]

    2.62 Seconds to search and locate 18 windows...
    1 cars found


     47%|████▋     | 592/1261 [36:45<40:45,  3.66s/it]

    2.58 Seconds to search and locate 20 windows...
    1 cars found


     47%|████▋     | 593/1261 [36:49<40:40,  3.65s/it]

    2.66 Seconds to search and locate 20 windows...
    1 cars found


     47%|████▋     | 594/1261 [36:52<40:39,  3.66s/it]

    2.66 Seconds to search and locate 22 windows...
    1 cars found


     47%|████▋     | 595/1261 [36:56<40:24,  3.64s/it]

    2.58 Seconds to search and locate 17 windows...
    1 cars found


     47%|████▋     | 596/1261 [37:00<40:20,  3.64s/it]

    2.61 Seconds to search and locate 17 windows...
    1 cars found


     47%|████▋     | 597/1261 [37:03<40:24,  3.65s/it]

    2.68 Seconds to search and locate 19 windows...
    1 cars found


     47%|████▋     | 598/1261 [37:07<40:17,  3.65s/it]

    2.61 Seconds to search and locate 25 windows...
    1 cars found


     48%|████▊     | 599/1261 [37:11<40:03,  3.63s/it]

    2.59 Seconds to search and locate 24 windows...
    1 cars found


     48%|████▊     | 600/1261 [37:14<39:59,  3.63s/it]

    2.64 Seconds to search and locate 27 windows...
    1 cars found


     48%|████▊     | 601/1261 [37:18<39:56,  3.63s/it]

    2.63 Seconds to search and locate 15 windows...
    1 cars found


     48%|████▊     | 602/1261 [37:21<39:47,  3.62s/it]

    2.6 Seconds to search and locate 15 windows...
    1 cars found


     48%|████▊     | 603/1261 [37:25<39:37,  3.61s/it]

    2.55 Seconds to search and locate 17 windows...
    1 cars found


     48%|████▊     | 604/1261 [37:29<39:29,  3.61s/it]

    2.61 Seconds to search and locate 18 windows...
    1 cars found


     48%|████▊     | 605/1261 [37:32<39:41,  3.63s/it]

    2.65 Seconds to search and locate 21 windows...
    1 cars found


     48%|████▊     | 606/1261 [37:36<40:00,  3.66s/it]

    2.74 Seconds to search and locate 18 windows...
    1 cars found


     48%|████▊     | 607/1261 [37:40<39:49,  3.65s/it]

    2.58 Seconds to search and locate 16 windows...
    1 cars found


     48%|████▊     | 608/1261 [37:43<39:53,  3.67s/it]

    2.69 Seconds to search and locate 20 windows...
    1 cars found


     48%|████▊     | 609/1261 [37:47<39:34,  3.64s/it]

    2.59 Seconds to search and locate 18 windows...
    1 cars found


     48%|████▊     | 610/1261 [37:51<39:19,  3.63s/it]

    2.57 Seconds to search and locate 17 windows...
    1 cars found


     48%|████▊     | 611/1261 [37:54<39:22,  3.63s/it]

    2.65 Seconds to search and locate 16 windows...
    1 cars found


     49%|████▊     | 612/1261 [37:58<39:22,  3.64s/it]

    2.67 Seconds to search and locate 14 windows...
    1 cars found


     49%|████▊     | 613/1261 [38:01<39:14,  3.63s/it]

    2.62 Seconds to search and locate 14 windows...
    1 cars found


     49%|████▊     | 614/1261 [38:05<39:21,  3.65s/it]

    2.61 Seconds to search and locate 16 windows...
    1 cars found


     49%|████▉     | 615/1261 [38:09<39:19,  3.65s/it]

    2.65 Seconds to search and locate 18 windows...
    1 cars found


     49%|████▉     | 616/1261 [38:12<39:16,  3.65s/it]

    2.65 Seconds to search and locate 19 windows...
    1 cars found


     49%|████▉     | 617/1261 [38:16<38:50,  3.62s/it]

    2.55 Seconds to search and locate 16 windows...
    1 cars found


     49%|████▉     | 618/1261 [38:20<38:51,  3.63s/it]

    2.6 Seconds to search and locate 14 windows...
    1 cars found


     49%|████▉     | 619/1261 [38:23<39:03,  3.65s/it]

    2.68 Seconds to search and locate 19 windows...
    1 cars found


     49%|████▉     | 620/1261 [38:27<38:55,  3.64s/it]

    2.62 Seconds to search and locate 15 windows...
    1 cars found


     49%|████▉     | 621/1261 [38:31<38:49,  3.64s/it]

    2.59 Seconds to search and locate 16 windows...
    1 cars found


     49%|████▉     | 622/1261 [38:34<38:48,  3.64s/it]

    2.64 Seconds to search and locate 18 windows...
    1 cars found


     49%|████▉     | 623/1261 [38:38<38:56,  3.66s/it]

    2.68 Seconds to search and locate 16 windows...
    1 cars found


     49%|████▉     | 624/1261 [38:42<38:54,  3.67s/it]

    2.67 Seconds to search and locate 14 windows...
    1 cars found


     50%|████▉     | 625/1261 [38:45<38:42,  3.65s/it]

    2.57 Seconds to search and locate 14 windows...
    1 cars found


     50%|████▉     | 626/1261 [38:49<38:44,  3.66s/it]

    2.67 Seconds to search and locate 16 windows...
    1 cars found


     50%|████▉     | 627/1261 [38:53<38:48,  3.67s/it]

    2.68 Seconds to search and locate 12 windows...
    1 cars found


     50%|████▉     | 628/1261 [38:56<38:28,  3.65s/it]

    2.6 Seconds to search and locate 14 windows...
    1 cars found


     50%|████▉     | 629/1261 [39:00<38:29,  3.65s/it]

    2.63 Seconds to search and locate 14 windows...
    1 cars found


     50%|████▉     | 630/1261 [39:04<38:22,  3.65s/it]

    2.64 Seconds to search and locate 11 windows...
    1 cars found


     50%|█████     | 631/1261 [39:07<38:16,  3.64s/it]

    2.62 Seconds to search and locate 2 windows...
    0 cars found


     50%|█████     | 632/1261 [39:11<38:10,  3.64s/it]

    2.6 Seconds to search and locate 3 windows...
    0 cars found


     50%|█████     | 633/1261 [39:14<38:06,  3.64s/it]

    2.62 Seconds to search and locate 6 windows...
    1 cars found


     50%|█████     | 634/1261 [39:18<37:56,  3.63s/it]

    2.62 Seconds to search and locate 5 windows...
    0 cars found


     50%|█████     | 635/1261 [39:22<37:50,  3.63s/it]

    2.6 Seconds to search and locate 5 windows...
    0 cars found


     50%|█████     | 636/1261 [39:25<37:46,  3.63s/it]

    2.57 Seconds to search and locate 7 windows...
    1 cars found


     51%|█████     | 637/1261 [39:29<37:50,  3.64s/it]

    2.67 Seconds to search and locate 9 windows...
    1 cars found


     51%|█████     | 638/1261 [39:33<37:56,  3.65s/it]

    2.68 Seconds to search and locate 9 windows...
    1 cars found


     51%|█████     | 639/1261 [39:36<37:37,  3.63s/it]

    2.6 Seconds to search and locate 12 windows...
    1 cars found


     51%|█████     | 640/1261 [39:40<37:36,  3.63s/it]

    2.61 Seconds to search and locate 13 windows...
    1 cars found


     51%|█████     | 641/1261 [39:43<37:35,  3.64s/it]

    2.67 Seconds to search and locate 17 windows...
    1 cars found


     51%|█████     | 642/1261 [39:47<37:20,  3.62s/it]

    2.62 Seconds to search and locate 14 windows...
    1 cars found


     51%|█████     | 643/1261 [39:51<37:07,  3.60s/it]

    2.59 Seconds to search and locate 16 windows...
    1 cars found


     51%|█████     | 644/1261 [39:54<37:06,  3.61s/it]

    2.64 Seconds to search and locate 10 windows...
    1 cars found


     51%|█████     | 645/1261 [39:58<37:15,  3.63s/it]

    2.7 Seconds to search and locate 16 windows...
    1 cars found


     51%|█████     | 646/1261 [40:02<37:09,  3.62s/it]

    2.62 Seconds to search and locate 12 windows...
    1 cars found


     51%|█████▏    | 647/1261 [40:05<37:13,  3.64s/it]

    2.62 Seconds to search and locate 16 windows...
    1 cars found


     51%|█████▏    | 648/1261 [40:09<37:14,  3.65s/it]

    2.68 Seconds to search and locate 14 windows...
    1 cars found


     51%|█████▏    | 649/1261 [40:13<37:14,  3.65s/it]

    2.67 Seconds to search and locate 10 windows...
    1 cars found


     52%|█████▏    | 650/1261 [40:16<36:57,  3.63s/it]

    2.58 Seconds to search and locate 10 windows...
    1 cars found


     52%|█████▏    | 651/1261 [40:20<36:57,  3.64s/it]

    2.61 Seconds to search and locate 16 windows...
    1 cars found


     52%|█████▏    | 652/1261 [40:23<36:57,  3.64s/it]

    2.68 Seconds to search and locate 12 windows...
    1 cars found


     52%|█████▏    | 653/1261 [40:27<36:55,  3.64s/it]

    2.66 Seconds to search and locate 11 windows...
    1 cars found


     52%|█████▏    | 654/1261 [40:31<36:42,  3.63s/it]

    2.61 Seconds to search and locate 13 windows...
    1 cars found


     52%|█████▏    | 655/1261 [40:34<36:42,  3.63s/it]

    2.64 Seconds to search and locate 9 windows...
    1 cars found


     52%|█████▏    | 656/1261 [40:38<36:52,  3.66s/it]

    2.7 Seconds to search and locate 11 windows...
    1 cars found


     52%|█████▏    | 657/1261 [40:42<36:47,  3.65s/it]

    2.67 Seconds to search and locate 12 windows...
    1 cars found


     52%|█████▏    | 658/1261 [40:45<36:33,  3.64s/it]

    2.57 Seconds to search and locate 11 windows...
    1 cars found


     52%|█████▏    | 659/1261 [40:49<36:34,  3.64s/it]

    2.68 Seconds to search and locate 13 windows...
    1 cars found


     52%|█████▏    | 660/1261 [40:53<36:38,  3.66s/it]

    2.7 Seconds to search and locate 10 windows...
    1 cars found


     52%|█████▏    | 661/1261 [40:56<36:08,  3.61s/it]

    2.53 Seconds to search and locate 5 windows...
    0 cars found


     52%|█████▏    | 662/1261 [41:00<36:16,  3.63s/it]

    2.65 Seconds to search and locate 5 windows...
    0 cars found


     53%|█████▎    | 663/1261 [41:03<36:12,  3.63s/it]

    2.66 Seconds to search and locate 6 windows...
    0 cars found


     53%|█████▎    | 664/1261 [41:07<36:02,  3.62s/it]

    2.61 Seconds to search and locate 7 windows...
    1 cars found


     53%|█████▎    | 665/1261 [41:11<35:49,  3.61s/it]

    2.57 Seconds to search and locate 6 windows...
    0 cars found


     53%|█████▎    | 666/1261 [41:14<35:53,  3.62s/it]

    2.65 Seconds to search and locate 11 windows...
    1 cars found


     53%|█████▎    | 667/1261 [41:18<35:55,  3.63s/it]

    2.68 Seconds to search and locate 11 windows...
    1 cars found


     53%|█████▎    | 668/1261 [41:21<35:39,  3.61s/it]

    2.56 Seconds to search and locate 11 windows...
    1 cars found


     53%|█████▎    | 669/1261 [41:25<35:36,  3.61s/it]

    2.58 Seconds to search and locate 14 windows...
    1 cars found


     53%|█████▎    | 670/1261 [41:29<35:43,  3.63s/it]

    2.68 Seconds to search and locate 16 windows...
    1 cars found


     53%|█████▎    | 671/1261 [41:32<35:50,  3.65s/it]

    2.7 Seconds to search and locate 14 windows...
    1 cars found


     53%|█████▎    | 672/1261 [41:36<35:38,  3.63s/it]

    2.62 Seconds to search and locate 15 windows...
    1 cars found


     53%|█████▎    | 673/1261 [41:40<35:38,  3.64s/it]

    2.61 Seconds to search and locate 13 windows...
    1 cars found


     53%|█████▎    | 674/1261 [41:43<35:39,  3.64s/it]

    2.68 Seconds to search and locate 7 windows...
    1 cars found


     54%|█████▎    | 675/1261 [41:47<35:25,  3.63s/it]

    2.61 Seconds to search and locate 9 windows...
    1 cars found


     54%|█████▎    | 676/1261 [41:51<35:26,  3.63s/it]

    2.61 Seconds to search and locate 13 windows...
    1 cars found


     54%|█████▎    | 677/1261 [41:54<35:31,  3.65s/it]

    2.7 Seconds to search and locate 7 windows...
    1 cars found


     54%|█████▍    | 678/1261 [41:58<35:29,  3.65s/it]

    2.69 Seconds to search and locate 10 windows...
    1 cars found


     54%|█████▍    | 679/1261 [42:02<35:18,  3.64s/it]

    2.63 Seconds to search and locate 7 windows...
    1 cars found


     54%|█████▍    | 680/1261 [42:05<35:17,  3.64s/it]

    2.6 Seconds to search and locate 5 windows...
    0 cars found


     54%|█████▍    | 681/1261 [42:09<35:15,  3.65s/it]

    2.69 Seconds to search and locate 6 windows...
    1 cars found


     54%|█████▍    | 682/1261 [42:13<35:18,  3.66s/it]

    2.69 Seconds to search and locate 7 windows...
    1 cars found


     54%|█████▍    | 683/1261 [42:16<34:50,  3.62s/it]

    2.55 Seconds to search and locate 4 windows...
    0 cars found


     54%|█████▍    | 684/1261 [42:20<34:51,  3.62s/it]

    2.6 Seconds to search and locate 5 windows...
    0 cars found


     54%|█████▍    | 685/1261 [42:23<34:53,  3.64s/it]

    2.68 Seconds to search and locate 4 windows...
    0 cars found


     54%|█████▍    | 686/1261 [42:27<34:44,  3.63s/it]

    2.65 Seconds to search and locate 5 windows...
    0 cars found


     54%|█████▍    | 687/1261 [42:31<34:35,  3.62s/it]

    2.59 Seconds to search and locate 7 windows...
    0 cars found


     55%|█████▍    | 688/1261 [42:34<34:35,  3.62s/it]

    2.64 Seconds to search and locate 10 windows...
    1 cars found


     55%|█████▍    | 689/1261 [42:38<34:44,  3.64s/it]

    2.65 Seconds to search and locate 12 windows...
    1 cars found


     55%|█████▍    | 690/1261 [42:42<34:54,  3.67s/it]

    2.69 Seconds to search and locate 10 windows...
    1 cars found


     55%|█████▍    | 691/1261 [42:45<34:40,  3.65s/it]

    2.58 Seconds to search and locate 6 windows...
    0 cars found


     55%|█████▍    | 692/1261 [42:49<34:42,  3.66s/it]

    2.71 Seconds to search and locate 11 windows...
    1 cars found


     55%|█████▍    | 693/1261 [42:53<34:49,  3.68s/it]

    2.73 Seconds to search and locate 14 windows...
    2 cars found


     55%|█████▌    | 694/1261 [42:56<34:26,  3.64s/it]

    2.58 Seconds to search and locate 21 windows...
    2 cars found


     55%|█████▌    | 695/1261 [43:00<34:34,  3.66s/it]

    2.63 Seconds to search and locate 15 windows...
    2 cars found


     55%|█████▌    | 696/1261 [43:04<34:34,  3.67s/it]

    2.69 Seconds to search and locate 14 windows...
    2 cars found


     55%|█████▌    | 697/1261 [43:07<34:30,  3.67s/it]

    2.7 Seconds to search and locate 18 windows...
    1 cars found


     55%|█████▌    | 698/1261 [43:11<34:07,  3.64s/it]

    2.58 Seconds to search and locate 21 windows...
    2 cars found


     55%|█████▌    | 699/1261 [43:14<34:07,  3.64s/it]

    2.63 Seconds to search and locate 17 windows...
    2 cars found


     56%|█████▌    | 700/1261 [43:18<34:09,  3.65s/it]

    2.69 Seconds to search and locate 26 windows...
    2 cars found


     56%|█████▌    | 701/1261 [43:22<34:00,  3.64s/it]

    2.62 Seconds to search and locate 23 windows...
    2 cars found


     56%|█████▌    | 702/1261 [43:25<33:51,  3.63s/it]

    2.57 Seconds to search and locate 28 windows...
    2 cars found


     56%|█████▌    | 703/1261 [43:29<34:02,  3.66s/it]

    2.72 Seconds to search and locate 33 windows...
    2 cars found


     56%|█████▌    | 704/1261 [43:33<34:12,  3.69s/it]

    2.73 Seconds to search and locate 36 windows...
    2 cars found


     56%|█████▌    | 705/1261 [43:36<33:55,  3.66s/it]

    2.62 Seconds to search and locate 41 windows...
    2 cars found


     56%|█████▌    | 706/1261 [43:40<33:55,  3.67s/it]

    2.62 Seconds to search and locate 64 windows...
    3 cars found


     56%|█████▌    | 707/1261 [43:44<34:02,  3.69s/it]

    2.72 Seconds to search and locate 77 windows...
    2 cars found


     56%|█████▌    | 708/1261 [43:48<34:03,  3.70s/it]

    2.71 Seconds to search and locate 78 windows...
    2 cars found


     56%|█████▌    | 709/1261 [43:51<33:48,  3.68s/it]

    2.62 Seconds to search and locate 71 windows...
    1 cars found


     56%|█████▋    | 710/1261 [43:55<33:42,  3.67s/it]

    2.62 Seconds to search and locate 74 windows...
    1 cars found


     56%|█████▋    | 711/1261 [43:59<33:44,  3.68s/it]

    2.71 Seconds to search and locate 61 windows...
    1 cars found


     56%|█████▋    | 712/1261 [44:02<33:48,  3.69s/it]

    2.7 Seconds to search and locate 65 windows...
    1 cars found


     57%|█████▋    | 713/1261 [44:06<33:40,  3.69s/it]

    2.65 Seconds to search and locate 78 windows...
    2 cars found


     57%|█████▋    | 714/1261 [44:10<33:39,  3.69s/it]

    2.66 Seconds to search and locate 66 windows...
    3 cars found


     57%|█████▋    | 715/1261 [44:13<33:39,  3.70s/it]

    2.73 Seconds to search and locate 61 windows...
    2 cars found


     57%|█████▋    | 716/1261 [44:17<33:27,  3.68s/it]

    2.65 Seconds to search and locate 53 windows...
    2 cars found


     57%|█████▋    | 717/1261 [44:21<33:21,  3.68s/it]

    2.63 Seconds to search and locate 70 windows...
    4 cars found


     57%|█████▋    | 718/1261 [44:24<33:19,  3.68s/it]

    2.67 Seconds to search and locate 63 windows...
    2 cars found


     57%|█████▋    | 719/1261 [44:28<33:18,  3.69s/it]

    2.72 Seconds to search and locate 75 windows...
    3 cars found


     57%|█████▋    | 720/1261 [44:32<33:10,  3.68s/it]

    2.65 Seconds to search and locate 84 windows...
    2 cars found


     57%|█████▋    | 721/1261 [44:35<33:04,  3.68s/it]

    2.62 Seconds to search and locate 60 windows...
    2 cars found


     57%|█████▋    | 722/1261 [44:39<33:09,  3.69s/it]

    2.71 Seconds to search and locate 58 windows...
    3 cars found


     57%|█████▋    | 723/1261 [44:43<33:10,  3.70s/it]

    2.73 Seconds to search and locate 70 windows...
    2 cars found


     57%|█████▋    | 724/1261 [44:46<32:47,  3.66s/it]

    2.58 Seconds to search and locate 81 windows...
    3 cars found


     57%|█████▋    | 725/1261 [44:50<32:44,  3.66s/it]

    2.61 Seconds to search and locate 97 windows...
    3 cars found


     58%|█████▊    | 726/1261 [44:54<32:46,  3.68s/it]

    2.7 Seconds to search and locate 103 windows...
    2 cars found


     58%|█████▊    | 727/1261 [44:57<32:44,  3.68s/it]

    2.71 Seconds to search and locate 87 windows...
    2 cars found


     58%|█████▊    | 728/1261 [45:01<32:30,  3.66s/it]

    2.6 Seconds to search and locate 87 windows...
    2 cars found


     58%|█████▊    | 729/1261 [45:05<32:27,  3.66s/it]

    2.62 Seconds to search and locate 87 windows...
    1 cars found


     58%|█████▊    | 730/1261 [45:08<32:32,  3.68s/it]

    2.69 Seconds to search and locate 102 windows...
    1 cars found


     58%|█████▊    | 731/1261 [45:12<32:30,  3.68s/it]

    2.7 Seconds to search and locate 109 windows...
    1 cars found


     58%|█████▊    | 732/1261 [45:16<32:12,  3.65s/it]

    2.57 Seconds to search and locate 99 windows...
    1 cars found


     58%|█████▊    | 733/1261 [45:19<32:18,  3.67s/it]

    2.72 Seconds to search and locate 99 windows...
    1 cars found


     58%|█████▊    | 734/1261 [45:23<32:18,  3.68s/it]

    2.68 Seconds to search and locate 96 windows...
    1 cars found


     58%|█████▊    | 735/1261 [45:27<32:02,  3.66s/it]

    2.62 Seconds to search and locate 117 windows...
    1 cars found


     58%|█████▊    | 736/1261 [45:30<32:05,  3.67s/it]

    2.66 Seconds to search and locate 123 windows...
    1 cars found


     58%|█████▊    | 737/1261 [45:34<32:02,  3.67s/it]

    2.68 Seconds to search and locate 140 windows...
    1 cars found


     59%|█████▊    | 738/1261 [45:38<32:08,  3.69s/it]

    2.72 Seconds to search and locate 142 windows...
    1 cars found


     59%|█████▊    | 739/1261 [45:41<31:54,  3.67s/it]

    2.63 Seconds to search and locate 140 windows...
    1 cars found


     59%|█████▊    | 740/1261 [45:45<31:55,  3.68s/it]

    2.63 Seconds to search and locate 148 windows...
    1 cars found


     59%|█████▉    | 741/1261 [45:49<31:57,  3.69s/it]

    2.73 Seconds to search and locate 143 windows...
    1 cars found


     59%|█████▉    | 742/1261 [45:53<31:53,  3.69s/it]

    2.67 Seconds to search and locate 137 windows...
    1 cars found


     59%|█████▉    | 743/1261 [45:56<31:36,  3.66s/it]

    2.61 Seconds to search and locate 137 windows...
    1 cars found


     59%|█████▉    | 744/1261 [46:00<31:43,  3.68s/it]

    2.68 Seconds to search and locate 141 windows...
    1 cars found


     59%|█████▉    | 745/1261 [46:04<31:42,  3.69s/it]

    2.7 Seconds to search and locate 139 windows...
    1 cars found


     59%|█████▉    | 746/1261 [46:07<31:37,  3.68s/it]

    2.69 Seconds to search and locate 144 windows...
    1 cars found


     59%|█████▉    | 747/1261 [46:11<31:28,  3.67s/it]

    2.63 Seconds to search and locate 137 windows...
    1 cars found


     59%|█████▉    | 748/1261 [46:15<31:24,  3.67s/it]

    2.6 Seconds to search and locate 139 windows...
    1 cars found


     59%|█████▉    | 749/1261 [46:18<31:20,  3.67s/it]

    2.72 Seconds to search and locate 133 windows...
    1 cars found


     59%|█████▉    | 750/1261 [46:22<31:10,  3.66s/it]

    2.64 Seconds to search and locate 135 windows...
    1 cars found


     60%|█████▉    | 751/1261 [46:26<30:59,  3.65s/it]

    2.57 Seconds to search and locate 118 windows...
    1 cars found


     60%|█████▉    | 752/1261 [46:29<31:01,  3.66s/it]

    2.68 Seconds to search and locate 120 windows...
    1 cars found


     60%|█████▉    | 753/1261 [46:33<31:06,  3.67s/it]

    2.71 Seconds to search and locate 114 windows...
    1 cars found


     60%|█████▉    | 754/1261 [46:37<30:55,  3.66s/it]

    2.64 Seconds to search and locate 117 windows...
    1 cars found


     60%|█████▉    | 755/1261 [46:40<31:10,  3.70s/it]

    2.7 Seconds to search and locate 117 windows...
    1 cars found


     60%|█████▉    | 756/1261 [46:44<31:10,  3.70s/it]

    2.71 Seconds to search and locate 124 windows...
    1 cars found


     60%|██████    | 757/1261 [46:48<31:01,  3.69s/it]

    2.69 Seconds to search and locate 127 windows...
    1 cars found


     60%|██████    | 758/1261 [46:51<30:50,  3.68s/it]

    2.64 Seconds to search and locate 128 windows...
    1 cars found


     60%|██████    | 759/1261 [46:55<30:46,  3.68s/it]

    2.63 Seconds to search and locate 126 windows...
    1 cars found


     60%|██████    | 760/1261 [46:59<30:47,  3.69s/it]

    2.7 Seconds to search and locate 133 windows...
    2 cars found


     60%|██████    | 761/1261 [47:02<30:45,  3.69s/it]

    2.7 Seconds to search and locate 122 windows...
    2 cars found


     60%|██████    | 762/1261 [47:06<30:26,  3.66s/it]

    2.6 Seconds to search and locate 126 windows...
    1 cars found


     61%|██████    | 763/1261 [47:10<30:25,  3.67s/it]

    2.66 Seconds to search and locate 121 windows...
    1 cars found


     61%|██████    | 764/1261 [47:13<30:22,  3.67s/it]

    2.69 Seconds to search and locate 117 windows...
    1 cars found


     61%|██████    | 765/1261 [47:17<30:09,  3.65s/it]

    2.62 Seconds to search and locate 115 windows...
    1 cars found


     61%|██████    | 766/1261 [47:21<30:03,  3.64s/it]

    2.61 Seconds to search and locate 107 windows...
    1 cars found


     61%|██████    | 767/1261 [47:24<30:03,  3.65s/it]

    2.67 Seconds to search and locate 119 windows...
    1 cars found


     61%|██████    | 768/1261 [47:28<30:07,  3.67s/it]

    2.72 Seconds to search and locate 125 windows...
    1 cars found


     61%|██████    | 769/1261 [47:32<30:02,  3.66s/it]

    2.63 Seconds to search and locate 116 windows...
    1 cars found


     61%|██████    | 770/1261 [47:35<29:56,  3.66s/it]

    2.59 Seconds to search and locate 106 windows...
    1 cars found


     61%|██████    | 771/1261 [47:39<30:04,  3.68s/it]

    2.7 Seconds to search and locate 116 windows...
    1 cars found


     61%|██████    | 772/1261 [47:43<30:12,  3.71s/it]

    2.76 Seconds to search and locate 111 windows...
    1 cars found


     61%|██████▏   | 773/1261 [47:46<29:54,  3.68s/it]

    2.59 Seconds to search and locate 110 windows...
    1 cars found


     61%|██████▏   | 774/1261 [47:50<29:50,  3.68s/it]

    2.63 Seconds to search and locate 117 windows...
    1 cars found


     61%|██████▏   | 775/1261 [47:54<29:49,  3.68s/it]

    2.68 Seconds to search and locate 101 windows...
    1 cars found


     62%|██████▏   | 776/1261 [47:58<29:48,  3.69s/it]

    2.71 Seconds to search and locate 95 windows...
    1 cars found


     62%|██████▏   | 777/1261 [48:01<29:35,  3.67s/it]

    2.62 Seconds to search and locate 95 windows...
    1 cars found


     62%|██████▏   | 778/1261 [48:05<29:32,  3.67s/it]

    2.63 Seconds to search and locate 92 windows...
    1 cars found


     62%|██████▏   | 779/1261 [48:09<29:35,  3.68s/it]

    2.71 Seconds to search and locate 88 windows...
    1 cars found


     62%|██████▏   | 780/1261 [48:12<29:32,  3.68s/it]

    2.69 Seconds to search and locate 91 windows...
    1 cars found


     62%|██████▏   | 781/1261 [48:16<29:23,  3.67s/it]

    2.6 Seconds to search and locate 95 windows...
    1 cars found


     62%|██████▏   | 782/1261 [48:20<29:19,  3.67s/it]

    2.67 Seconds to search and locate 95 windows...
    1 cars found


     62%|██████▏   | 783/1261 [48:23<29:15,  3.67s/it]

    2.69 Seconds to search and locate 88 windows...
    1 cars found


     62%|██████▏   | 784/1261 [48:27<29:02,  3.65s/it]

    2.61 Seconds to search and locate 90 windows...
    1 cars found


     62%|██████▏   | 785/1261 [48:30<29:01,  3.66s/it]

    2.61 Seconds to search and locate 90 windows...
    1 cars found


     62%|██████▏   | 786/1261 [48:34<29:03,  3.67s/it]

    2.71 Seconds to search and locate 95 windows...
    1 cars found


     62%|██████▏   | 787/1261 [48:38<29:05,  3.68s/it]

    2.71 Seconds to search and locate 98 windows...
    1 cars found


     62%|██████▏   | 788/1261 [48:42<29:01,  3.68s/it]

    2.65 Seconds to search and locate 86 windows...
    1 cars found


     63%|██████▎   | 789/1261 [48:45<28:53,  3.67s/it]

    2.59 Seconds to search and locate 81 windows...
    1 cars found


     63%|██████▎   | 790/1261 [48:49<28:50,  3.68s/it]

    2.69 Seconds to search and locate 84 windows...
    1 cars found


     63%|██████▎   | 791/1261 [48:53<28:52,  3.69s/it]

    2.71 Seconds to search and locate 85 windows...
    1 cars found


     63%|██████▎   | 792/1261 [48:56<28:38,  3.66s/it]

    2.61 Seconds to search and locate 88 windows...
    1 cars found


     63%|██████▎   | 793/1261 [49:00<28:39,  3.67s/it]

    2.66 Seconds to search and locate 84 windows...
    1 cars found


     63%|██████▎   | 794/1261 [49:04<28:39,  3.68s/it]

    2.69 Seconds to search and locate 84 windows...
    1 cars found


     63%|██████▎   | 795/1261 [49:07<28:36,  3.68s/it]

    2.7 Seconds to search and locate 87 windows...
    1 cars found


     63%|██████▎   | 796/1261 [49:11<28:27,  3.67s/it]

    2.64 Seconds to search and locate 85 windows...
    1 cars found


     63%|██████▎   | 797/1261 [49:15<28:21,  3.67s/it]

    2.63 Seconds to search and locate 82 windows...
    1 cars found


     63%|██████▎   | 798/1261 [49:18<28:16,  3.66s/it]

    2.67 Seconds to search and locate 85 windows...
    1 cars found


     63%|██████▎   | 799/1261 [49:22<28:06,  3.65s/it]

    2.61 Seconds to search and locate 80 windows...
    1 cars found


     63%|██████▎   | 800/1261 [49:26<28:00,  3.64s/it]

    2.58 Seconds to search and locate 86 windows...
    1 cars found


     64%|██████▎   | 801/1261 [49:29<28:06,  3.67s/it]

    2.72 Seconds to search and locate 85 windows...
    1 cars found


     64%|██████▎   | 802/1261 [49:33<28:11,  3.68s/it]

    2.72 Seconds to search and locate 86 windows...
    1 cars found


     64%|██████▎   | 803/1261 [49:37<27:53,  3.65s/it]

    2.59 Seconds to search and locate 84 windows...
    1 cars found


     64%|██████▍   | 804/1261 [49:40<27:56,  3.67s/it]

    2.61 Seconds to search and locate 95 windows...
    1 cars found


     64%|██████▍   | 805/1261 [49:44<27:56,  3.68s/it]

    2.7 Seconds to search and locate 86 windows...
    1 cars found


     64%|██████▍   | 806/1261 [49:48<27:51,  3.67s/it]

    2.68 Seconds to search and locate 89 windows...
    1 cars found


     64%|██████▍   | 807/1261 [49:51<27:39,  3.66s/it]

    2.61 Seconds to search and locate 86 windows...
    1 cars found


     64%|██████▍   | 808/1261 [49:55<27:35,  3.65s/it]

    2.61 Seconds to search and locate 88 windows...
    1 cars found


     64%|██████▍   | 809/1261 [49:59<27:39,  3.67s/it]

    2.7 Seconds to search and locate 85 windows...
    1 cars found


     64%|██████▍   | 810/1261 [50:02<27:37,  3.68s/it]

    2.67 Seconds to search and locate 78 windows...
    1 cars found


     64%|██████▍   | 811/1261 [50:06<27:23,  3.65s/it]

    2.6 Seconds to search and locate 74 windows...
    1 cars found


     64%|██████▍   | 812/1261 [50:10<27:25,  3.67s/it]

    2.64 Seconds to search and locate 70 windows...
    1 cars found


     64%|██████▍   | 813/1261 [50:13<27:26,  3.68s/it]

    2.69 Seconds to search and locate 71 windows...
    1 cars found


     65%|██████▍   | 814/1261 [50:17<27:10,  3.65s/it]

    2.6 Seconds to search and locate 75 windows...
    1 cars found


     65%|██████▍   | 815/1261 [50:21<27:10,  3.66s/it]

    2.63 Seconds to search and locate 75 windows...
    1 cars found


     65%|██████▍   | 816/1261 [50:24<27:09,  3.66s/it]

    2.68 Seconds to search and locate 74 windows...
    1 cars found


     65%|██████▍   | 817/1261 [50:28<27:09,  3.67s/it]

    2.7 Seconds to search and locate 74 windows...
    1 cars found


     65%|██████▍   | 818/1261 [50:31<26:58,  3.65s/it]

    2.62 Seconds to search and locate 69 windows...
    1 cars found


     65%|██████▍   | 819/1261 [50:35<26:54,  3.65s/it]

    2.58 Seconds to search and locate 76 windows...
    1 cars found


     65%|██████▌   | 820/1261 [50:39<27:00,  3.67s/it]

    2.72 Seconds to search and locate 81 windows...
    1 cars found


     65%|██████▌   | 821/1261 [50:43<27:00,  3.68s/it]

    2.7 Seconds to search and locate 76 windows...
    1 cars found


     65%|██████▌   | 822/1261 [50:46<26:39,  3.64s/it]

    2.57 Seconds to search and locate 71 windows...
    1 cars found


     65%|██████▌   | 823/1261 [50:50<26:41,  3.66s/it]

    2.64 Seconds to search and locate 77 windows...
    1 cars found


     65%|██████▌   | 824/1261 [50:53<26:39,  3.66s/it]

    2.69 Seconds to search and locate 75 windows...
    1 cars found


     65%|██████▌   | 825/1261 [50:57<26:33,  3.65s/it]

    2.65 Seconds to search and locate 70 windows...
    1 cars found


     66%|██████▌   | 826/1261 [51:01<26:31,  3.66s/it]

    2.65 Seconds to search and locate 71 windows...
    1 cars found


     66%|██████▌   | 827/1261 [51:04<26:30,  3.66s/it]

    2.65 Seconds to search and locate 65 windows...
    1 cars found


     66%|██████▌   | 828/1261 [51:08<26:32,  3.68s/it]

    2.72 Seconds to search and locate 65 windows...
    1 cars found


     66%|██████▌   | 829/1261 [51:12<26:27,  3.68s/it]

    2.66 Seconds to search and locate 69 windows...
    1 cars found


     66%|██████▌   | 830/1261 [51:15<26:18,  3.66s/it]

    2.59 Seconds to search and locate 76 windows...
    1 cars found


     66%|██████▌   | 831/1261 [51:19<26:23,  3.68s/it]

    2.73 Seconds to search and locate 78 windows...
    1 cars found


     66%|██████▌   | 832/1261 [51:23<26:24,  3.69s/it]

    2.71 Seconds to search and locate 69 windows...
    1 cars found


     66%|██████▌   | 833/1261 [51:26<26:06,  3.66s/it]

    2.58 Seconds to search and locate 74 windows...
    1 cars found


     66%|██████▌   | 834/1261 [51:30<26:10,  3.68s/it]

    2.66 Seconds to search and locate 69 windows...
    1 cars found


     66%|██████▌   | 835/1261 [51:34<26:07,  3.68s/it]

    2.69 Seconds to search and locate 73 windows...
    1 cars found


     66%|██████▋   | 836/1261 [51:38<26:06,  3.69s/it]

    2.71 Seconds to search and locate 74 windows...
    1 cars found


     66%|██████▋   | 837/1261 [51:41<25:53,  3.66s/it]

    2.61 Seconds to search and locate 74 windows...
    1 cars found


     66%|██████▋   | 838/1261 [51:45<25:52,  3.67s/it]

    2.65 Seconds to search and locate 72 windows...
    1 cars found


     67%|██████▋   | 839/1261 [51:49<25:48,  3.67s/it]

    2.68 Seconds to search and locate 70 windows...
    1 cars found


     67%|██████▋   | 840/1261 [51:52<25:42,  3.66s/it]

    2.64 Seconds to search and locate 75 windows...
    1 cars found


     67%|██████▋   | 841/1261 [51:56<25:32,  3.65s/it]

    2.58 Seconds to search and locate 71 windows...
    1 cars found


     67%|██████▋   | 842/1261 [52:00<25:38,  3.67s/it]

    2.7 Seconds to search and locate 73 windows...
    1 cars found


     67%|██████▋   | 843/1261 [52:03<25:49,  3.71s/it]

    2.71 Seconds to search and locate 76 windows...
    1 cars found


     67%|██████▋   | 844/1261 [52:07<25:35,  3.68s/it]

    2.62 Seconds to search and locate 83 windows...
    1 cars found


     67%|██████▋   | 845/1261 [52:11<25:27,  3.67s/it]

    2.6 Seconds to search and locate 78 windows...
    1 cars found


     67%|██████▋   | 846/1261 [52:14<25:22,  3.67s/it]

    2.67 Seconds to search and locate 75 windows...
    1 cars found


     67%|██████▋   | 847/1261 [52:18<25:18,  3.67s/it]

    2.68 Seconds to search and locate 77 windows...
    1 cars found


     67%|██████▋   | 848/1261 [52:22<25:06,  3.65s/it]

    2.61 Seconds to search and locate 80 windows...
    1 cars found


     67%|██████▋   | 849/1261 [52:25<25:04,  3.65s/it]

    2.6 Seconds to search and locate 79 windows...
    1 cars found


     67%|██████▋   | 850/1261 [52:29<25:09,  3.67s/it]

    2.73 Seconds to search and locate 76 windows...
    1 cars found


     67%|██████▋   | 851/1261 [52:33<25:14,  3.69s/it]

    2.7 Seconds to search and locate 75 windows...
    1 cars found


     68%|██████▊   | 852/1261 [52:36<24:56,  3.66s/it]

    2.57 Seconds to search and locate 81 windows...
    1 cars found


     68%|██████▊   | 853/1261 [52:40<25:00,  3.68s/it]

    2.66 Seconds to search and locate 78 windows...
    1 cars found


     68%|██████▊   | 854/1261 [52:44<25:05,  3.70s/it]

    2.71 Seconds to search and locate 81 windows...
    1 cars found


     68%|██████▊   | 855/1261 [52:47<24:58,  3.69s/it]

    2.65 Seconds to search and locate 85 windows...
    1 cars found


     68%|██████▊   | 856/1261 [52:51<24:52,  3.68s/it]

    2.64 Seconds to search and locate 85 windows...
    1 cars found


     68%|██████▊   | 857/1261 [52:55<24:46,  3.68s/it]

    2.63 Seconds to search and locate 86 windows...
    1 cars found


     68%|██████▊   | 858/1261 [52:58<24:45,  3.68s/it]

    2.7 Seconds to search and locate 74 windows...
    1 cars found


     68%|██████▊   | 859/1261 [53:02<24:39,  3.68s/it]

    2.67 Seconds to search and locate 78 windows...
    1 cars found


     68%|██████▊   | 860/1261 [53:06<24:27,  3.66s/it]

    2.58 Seconds to search and locate 80 windows...
    1 cars found


     68%|██████▊   | 861/1261 [53:09<24:31,  3.68s/it]

    2.73 Seconds to search and locate 75 windows...
    1 cars found


     68%|██████▊   | 862/1261 [53:13<24:34,  3.69s/it]

    2.7 Seconds to search and locate 75 windows...
    1 cars found


     68%|██████▊   | 863/1261 [53:17<24:18,  3.66s/it]

    2.59 Seconds to search and locate 79 windows...
    1 cars found


     69%|██████▊   | 864/1261 [53:20<24:13,  3.66s/it]

    2.59 Seconds to search and locate 86 windows...
    1 cars found


     69%|██████▊   | 865/1261 [53:24<24:11,  3.66s/it]

    2.67 Seconds to search and locate 83 windows...
    1 cars found


     69%|██████▊   | 866/1261 [53:28<24:08,  3.67s/it]

    2.69 Seconds to search and locate 83 windows...
    1 cars found


     69%|██████▉   | 867/1261 [53:31<23:59,  3.65s/it]

    2.63 Seconds to search and locate 88 windows...
    1 cars found


     69%|██████▉   | 868/1261 [53:35<23:58,  3.66s/it]

    2.61 Seconds to search and locate 97 windows...
    2 cars found


     69%|██████▉   | 869/1261 [53:39<23:59,  3.67s/it]

    2.71 Seconds to search and locate 91 windows...
    1 cars found


     69%|██████▉   | 870/1261 [53:42<24:01,  3.69s/it]

    2.69 Seconds to search and locate 93 windows...
    1 cars found


     69%|██████▉   | 871/1261 [53:46<23:45,  3.66s/it]

    2.58 Seconds to search and locate 87 windows...
    1 cars found


     69%|██████▉   | 872/1261 [53:50<23:43,  3.66s/it]

    2.64 Seconds to search and locate 78 windows...
    1 cars found


     69%|██████▉   | 873/1261 [53:53<23:41,  3.66s/it]

    2.68 Seconds to search and locate 79 windows...
    1 cars found


     69%|██████▉   | 874/1261 [53:57<23:33,  3.65s/it]

    2.65 Seconds to search and locate 76 windows...
    1 cars found


     69%|██████▉   | 875/1261 [54:01<23:26,  3.64s/it]

    2.59 Seconds to search and locate 80 windows...
    1 cars found


     69%|██████▉   | 876/1261 [54:04<23:26,  3.65s/it]

    2.67 Seconds to search and locate 84 windows...
    1 cars found


     70%|██████▉   | 877/1261 [54:08<23:23,  3.65s/it]

    2.68 Seconds to search and locate 86 windows...
    1 cars found


     70%|██████▉   | 878/1261 [54:12<23:16,  3.65s/it]

    2.6 Seconds to search and locate 88 windows...
    1 cars found


     70%|██████▉   | 879/1261 [54:15<23:11,  3.64s/it]

    2.57 Seconds to search and locate 93 windows...
    1 cars found


     70%|██████▉   | 880/1261 [54:19<23:14,  3.66s/it]

    2.7 Seconds to search and locate 94 windows...
    1 cars found


     70%|██████▉   | 881/1261 [54:23<23:18,  3.68s/it]

    2.7 Seconds to search and locate 94 windows...
    1 cars found


     70%|██████▉   | 882/1261 [54:26<23:06,  3.66s/it]

    2.6 Seconds to search and locate 101 windows...
    1 cars found


     70%|███████   | 883/1261 [54:30<23:17,  3.70s/it]

    2.73 Seconds to search and locate 95 windows...
    1 cars found


     70%|███████   | 884/1261 [54:34<23:12,  3.69s/it]

    2.69 Seconds to search and locate 94 windows...
    1 cars found


     70%|███████   | 885/1261 [54:37<23:03,  3.68s/it]

    2.64 Seconds to search and locate 96 windows...
    1 cars found


     70%|███████   | 886/1261 [54:41<23:04,  3.69s/it]

    2.65 Seconds to search and locate 93 windows...
    1 cars found


     70%|███████   | 887/1261 [54:45<22:56,  3.68s/it]

    2.63 Seconds to search and locate 93 windows...
    1 cars found


     70%|███████   | 888/1261 [54:48<22:52,  3.68s/it]

    2.69 Seconds to search and locate 90 windows...
    1 cars found


     70%|███████   | 889/1261 [54:52<22:44,  3.67s/it]

    2.62 Seconds to search and locate 94 windows...
    1 cars found


     71%|███████   | 890/1261 [54:56<22:35,  3.65s/it]

    2.58 Seconds to search and locate 93 windows...
    1 cars found


     71%|███████   | 891/1261 [54:59<22:36,  3.67s/it]

    2.69 Seconds to search and locate 98 windows...
    1 cars found


     71%|███████   | 892/1261 [55:03<22:37,  3.68s/it]

    2.69 Seconds to search and locate 105 windows...
    1 cars found


     71%|███████   | 893/1261 [55:07<22:20,  3.64s/it]

    2.56 Seconds to search and locate 103 windows...
    1 cars found


     71%|███████   | 894/1261 [55:10<22:24,  3.66s/it]

    2.64 Seconds to search and locate 102 windows...
    1 cars found


     71%|███████   | 895/1261 [55:14<22:24,  3.67s/it]

    2.68 Seconds to search and locate 102 windows...
    1 cars found


     71%|███████   | 896/1261 [55:18<22:22,  3.68s/it]

    2.68 Seconds to search and locate 100 windows...
    1 cars found


     71%|███████   | 897/1261 [55:21<22:14,  3.67s/it]

    2.62 Seconds to search and locate 102 windows...
    1 cars found


     71%|███████   | 898/1261 [55:25<22:11,  3.67s/it]

    2.61 Seconds to search and locate 103 windows...
    1 cars found


     71%|███████▏  | 899/1261 [55:29<22:11,  3.68s/it]

    2.71 Seconds to search and locate 99 windows...
    1 cars found


     71%|███████▏  | 900/1261 [55:32<22:07,  3.68s/it]

    2.66 Seconds to search and locate 106 windows...
    1 cars found


     71%|███████▏  | 901/1261 [55:36<21:56,  3.66s/it]

    2.59 Seconds to search and locate 99 windows...
    1 cars found


     72%|███████▏  | 902/1261 [55:40<21:59,  3.68s/it]

    2.68 Seconds to search and locate 104 windows...
    1 cars found


     72%|███████▏  | 903/1261 [55:44<22:00,  3.69s/it]

    2.69 Seconds to search and locate 98 windows...
    1 cars found


     72%|███████▏  | 904/1261 [55:47<21:48,  3.66s/it]

    2.6 Seconds to search and locate 103 windows...
    1 cars found


     72%|███████▏  | 905/1261 [55:51<21:43,  3.66s/it]

    2.61 Seconds to search and locate 106 windows...
    1 cars found


     72%|███████▏  | 906/1261 [55:54<21:41,  3.67s/it]

    2.67 Seconds to search and locate 102 windows...
    1 cars found


     72%|███████▏  | 907/1261 [55:58<21:40,  3.67s/it]

    2.69 Seconds to search and locate 109 windows...
    1 cars found


     72%|███████▏  | 908/1261 [56:02<21:33,  3.66s/it]

    2.62 Seconds to search and locate 101 windows...
    1 cars found


     72%|███████▏  | 909/1261 [56:05<21:27,  3.66s/it]

    2.57 Seconds to search and locate 105 windows...
    1 cars found


     72%|███████▏  | 910/1261 [56:09<21:26,  3.67s/it]

    2.69 Seconds to search and locate 98 windows...
    1 cars found


     72%|███████▏  | 911/1261 [56:13<21:28,  3.68s/it]

    2.71 Seconds to search and locate 100 windows...
    1 cars found


     72%|███████▏  | 912/1261 [56:16<21:12,  3.65s/it]

    2.57 Seconds to search and locate 102 windows...
    1 cars found


     72%|███████▏  | 913/1261 [56:20<21:11,  3.65s/it]

    2.61 Seconds to search and locate 105 windows...
    1 cars found


     72%|███████▏  | 914/1261 [56:24<21:09,  3.66s/it]

    2.68 Seconds to search and locate 110 windows...
    1 cars found


     73%|███████▎  | 915/1261 [56:27<21:03,  3.65s/it]

    2.63 Seconds to search and locate 113 windows...
    1 cars found


     73%|███████▎  | 916/1261 [56:31<20:59,  3.65s/it]

    2.62 Seconds to search and locate 115 windows...
    1 cars found


     73%|███████▎  | 917/1261 [56:35<20:56,  3.65s/it]

    2.63 Seconds to search and locate 112 windows...
    1 cars found


     73%|███████▎  | 918/1261 [56:38<21:00,  3.67s/it]

    2.72 Seconds to search and locate 118 windows...
    1 cars found


     73%|███████▎  | 919/1261 [56:42<20:53,  3.67s/it]

    2.64 Seconds to search and locate 110 windows...
    1 cars found


     73%|███████▎  | 920/1261 [56:46<20:47,  3.66s/it]

    2.58 Seconds to search and locate 106 windows...
    1 cars found


     73%|███████▎  | 921/1261 [56:49<20:47,  3.67s/it]

    2.7 Seconds to search and locate 108 windows...
    1 cars found


     73%|███████▎  | 922/1261 [56:53<20:47,  3.68s/it]

    2.7 Seconds to search and locate 114 windows...
    1 cars found


     73%|███████▎  | 923/1261 [56:57<20:33,  3.65s/it]

    2.57 Seconds to search and locate 110 windows...
    1 cars found


     73%|███████▎  | 924/1261 [57:00<20:33,  3.66s/it]

    2.61 Seconds to search and locate 114 windows...
    1 cars found


     73%|███████▎  | 925/1261 [57:04<20:36,  3.68s/it]

    2.69 Seconds to search and locate 108 windows...
    1 cars found


     73%|███████▎  | 926/1261 [57:08<20:32,  3.68s/it]

    2.67 Seconds to search and locate 112 windows...
    1 cars found


     74%|███████▎  | 927/1261 [57:11<20:25,  3.67s/it]

    2.61 Seconds to search and locate 107 windows...
    1 cars found


     74%|███████▎  | 928/1261 [57:15<20:22,  3.67s/it]

    2.62 Seconds to search and locate 110 windows...
    1 cars found


     74%|███████▎  | 929/1261 [57:19<20:20,  3.68s/it]

    2.69 Seconds to search and locate 112 windows...
    1 cars found


     74%|███████▍  | 930/1261 [57:22<20:22,  3.69s/it]

    2.7 Seconds to search and locate 108 windows...
    1 cars found


     74%|███████▍  | 931/1261 [57:26<20:09,  3.67s/it]

    2.59 Seconds to search and locate 100 windows...
    1 cars found


     74%|███████▍  | 932/1261 [57:30<20:09,  3.67s/it]

    2.66 Seconds to search and locate 103 windows...
    1 cars found


     74%|███████▍  | 933/1261 [57:33<20:08,  3.68s/it]

    2.7 Seconds to search and locate 101 windows...
    1 cars found


     74%|███████▍  | 934/1261 [57:37<19:56,  3.66s/it]

    2.61 Seconds to search and locate 107 windows...
    1 cars found


     74%|███████▍  | 935/1261 [57:41<19:55,  3.67s/it]

    2.64 Seconds to search and locate 105 windows...
    1 cars found


     74%|███████▍  | 936/1261 [57:44<19:54,  3.68s/it]

    2.68 Seconds to search and locate 109 windows...
    1 cars found


     74%|███████▍  | 937/1261 [57:48<19:51,  3.68s/it]

    2.68 Seconds to search and locate 102 windows...
    1 cars found


     74%|███████▍  | 938/1261 [57:52<19:46,  3.67s/it]

    2.63 Seconds to search and locate 106 windows...
    1 cars found


     74%|███████▍  | 939/1261 [57:55<19:39,  3.66s/it]

    2.57 Seconds to search and locate 104 windows...
    1 cars found


     75%|███████▍  | 940/1261 [57:59<19:40,  3.68s/it]

    2.69 Seconds to search and locate 106 windows...
    2 cars found


     75%|███████▍  | 941/1261 [58:03<19:39,  3.69s/it]

    2.7 Seconds to search and locate 105 windows...
    1 cars found


     75%|███████▍  | 942/1261 [58:06<19:26,  3.66s/it]

    2.55 Seconds to search and locate 105 windows...
    2 cars found


     75%|███████▍  | 943/1261 [58:10<19:29,  3.68s/it]

    2.67 Seconds to search and locate 111 windows...
    1 cars found


     75%|███████▍  | 944/1261 [58:14<19:25,  3.68s/it]

    2.67 Seconds to search and locate 109 windows...
    1 cars found


     75%|███████▍  | 945/1261 [58:18<19:18,  3.67s/it]

    2.64 Seconds to search and locate 102 windows...
    2 cars found


     75%|███████▌  | 946/1261 [58:21<19:12,  3.66s/it]

    2.62 Seconds to search and locate 107 windows...
    2 cars found


     75%|███████▌  | 947/1261 [58:25<19:09,  3.66s/it]

    2.61 Seconds to search and locate 106 windows...
    1 cars found


     75%|███████▌  | 948/1261 [58:29<19:08,  3.67s/it]

    2.68 Seconds to search and locate 112 windows...
    1 cars found


     75%|███████▌  | 949/1261 [58:32<19:06,  3.67s/it]

    2.66 Seconds to search and locate 121 windows...
    1 cars found


     75%|███████▌  | 950/1261 [58:36<18:56,  3.66s/it]

    2.55 Seconds to search and locate 112 windows...
    1 cars found


     75%|███████▌  | 951/1261 [58:40<18:58,  3.67s/it]

    2.68 Seconds to search and locate 106 windows...
    2 cars found


     75%|███████▌  | 952/1261 [58:43<19:04,  3.70s/it]

    2.75 Seconds to search and locate 108 windows...
    2 cars found


     76%|███████▌  | 953/1261 [58:47<18:52,  3.68s/it]

    2.59 Seconds to search and locate 104 windows...
    2 cars found


     76%|███████▌  | 954/1261 [58:51<18:47,  3.67s/it]

    2.6 Seconds to search and locate 116 windows...
    2 cars found


     76%|███████▌  | 955/1261 [58:54<18:46,  3.68s/it]

    2.7 Seconds to search and locate 105 windows...
    2 cars found


     76%|███████▌  | 956/1261 [58:58<18:43,  3.68s/it]

    2.69 Seconds to search and locate 103 windows...
    2 cars found


     76%|███████▌  | 957/1261 [59:02<18:37,  3.67s/it]

    2.61 Seconds to search and locate 111 windows...
    1 cars found


     76%|███████▌  | 958/1261 [59:05<18:30,  3.67s/it]

    2.57 Seconds to search and locate 108 windows...
    1 cars found


     76%|███████▌  | 959/1261 [59:09<18:30,  3.68s/it]

    2.7 Seconds to search and locate 102 windows...
    2 cars found


     76%|███████▌  | 960/1261 [59:13<18:32,  3.70s/it]

    2.71 Seconds to search and locate 109 windows...
    2 cars found


     76%|███████▌  | 961/1261 [59:16<18:21,  3.67s/it]

    2.57 Seconds to search and locate 101 windows...
    2 cars found


     76%|███████▋  | 962/1261 [59:20<18:20,  3.68s/it]

    2.63 Seconds to search and locate 100 windows...
    2 cars found


     76%|███████▋  | 963/1261 [59:24<18:20,  3.69s/it]

    2.71 Seconds to search and locate 105 windows...
    2 cars found


     76%|███████▋  | 964/1261 [59:27<18:09,  3.67s/it]

    2.61 Seconds to search and locate 100 windows...
    2 cars found


     77%|███████▋  | 965/1261 [59:31<18:05,  3.67s/it]

    2.63 Seconds to search and locate 105 windows...
    2 cars found


     77%|███████▋  | 966/1261 [59:35<18:06,  3.68s/it]

    2.66 Seconds to search and locate 104 windows...
    2 cars found


     77%|███████▋  | 967/1261 [59:38<18:10,  3.71s/it]

    2.75 Seconds to search and locate 98 windows...
    2 cars found


     77%|███████▋  | 968/1261 [59:42<18:04,  3.70s/it]

    2.64 Seconds to search and locate 98 windows...
    2 cars found


     77%|███████▋  | 969/1261 [59:46<17:56,  3.69s/it]

    2.58 Seconds to search and locate 104 windows...
    2 cars found


     77%|███████▋  | 970/1261 [59:50<17:58,  3.70s/it]

    2.72 Seconds to search and locate 101 windows...
    2 cars found


     77%|███████▋  | 971/1261 [59:53<17:57,  3.72s/it]

    2.72 Seconds to search and locate 106 windows...
    2 cars found


     77%|███████▋  | 972/1261 [59:57<17:46,  3.69s/it]

    2.62 Seconds to search and locate 107 windows...
    2 cars found


     77%|███████▋  | 973/1261 [1:00:01<17:42,  3.69s/it]

    2.61 Seconds to search and locate 109 windows...
    2 cars found


     77%|███████▋  | 974/1261 [1:00:04<17:41,  3.70s/it]

    2.69 Seconds to search and locate 103 windows...
    2 cars found


     77%|███████▋  | 975/1261 [1:00:08<17:37,  3.70s/it]

    2.68 Seconds to search and locate 110 windows...
    2 cars found


     77%|███████▋  | 976/1261 [1:00:12<17:26,  3.67s/it]

    2.57 Seconds to search and locate 105 windows...
    2 cars found


     77%|███████▋  | 977/1261 [1:00:15<17:23,  3.67s/it]

    2.59 Seconds to search and locate 103 windows...
    2 cars found


     78%|███████▊  | 978/1261 [1:00:19<17:23,  3.69s/it]

    2.69 Seconds to search and locate 97 windows...
    2 cars found


     78%|███████▊  | 979/1261 [1:00:23<17:26,  3.71s/it]

    2.73 Seconds to search and locate 95 windows...
    2 cars found


     78%|███████▊  | 980/1261 [1:00:26<17:13,  3.68s/it]

    2.57 Seconds to search and locate 97 windows...
    2 cars found


     78%|███████▊  | 981/1261 [1:00:30<17:14,  3.69s/it]

    2.66 Seconds to search and locate 100 windows...
    2 cars found


     78%|███████▊  | 982/1261 [1:00:34<17:13,  3.70s/it]

    2.71 Seconds to search and locate 107 windows...
    2 cars found


     78%|███████▊  | 983/1261 [1:00:38<17:05,  3.69s/it]

    2.62 Seconds to search and locate 113 windows...
    2 cars found


     78%|███████▊  | 984/1261 [1:00:41<16:59,  3.68s/it]

    2.61 Seconds to search and locate 113 windows...
    2 cars found


     78%|███████▊  | 985/1261 [1:00:45<16:58,  3.69s/it]

    2.66 Seconds to search and locate 119 windows...
    2 cars found


     78%|███████▊  | 986/1261 [1:00:49<16:56,  3.69s/it]

    2.7 Seconds to search and locate 116 windows...
    2 cars found


     78%|███████▊  | 987/1261 [1:00:52<16:49,  3.68s/it]

    2.64 Seconds to search and locate 114 windows...
    2 cars found


     78%|███████▊  | 988/1261 [1:00:56<16:41,  3.67s/it]

    2.58 Seconds to search and locate 104 windows...
    3 cars found


     78%|███████▊  | 989/1261 [1:01:00<16:42,  3.69s/it]

    2.73 Seconds to search and locate 110 windows...
    2 cars found


     79%|███████▊  | 990/1261 [1:01:03<16:40,  3.69s/it]

    2.7 Seconds to search and locate 129 windows...
    2 cars found


     79%|███████▊  | 991/1261 [1:01:07<16:25,  3.65s/it]

    2.56 Seconds to search and locate 94 windows...
    2 cars found


     79%|███████▊  | 992/1261 [1:01:11<16:24,  3.66s/it]

    2.63 Seconds to search and locate 111 windows...
    2 cars found


     79%|███████▊  | 993/1261 [1:01:14<16:22,  3.67s/it]

    2.69 Seconds to search and locate 122 windows...
    2 cars found


     79%|███████▉  | 994/1261 [1:01:18<16:19,  3.67s/it]

    2.69 Seconds to search and locate 111 windows...
    2 cars found


     79%|███████▉  | 995/1261 [1:01:22<16:08,  3.64s/it]

    2.59 Seconds to search and locate 118 windows...
    2 cars found


     79%|███████▉  | 996/1261 [1:01:25<16:04,  3.64s/it]

    2.6 Seconds to search and locate 120 windows...
    2 cars found


     79%|███████▉  | 997/1261 [1:01:29<16:05,  3.66s/it]

    2.72 Seconds to search and locate 116 windows...
    2 cars found


     79%|███████▉  | 998/1261 [1:01:32<16:00,  3.65s/it]

    2.66 Seconds to search and locate 109 windows...
    3 cars found


     79%|███████▉  | 999/1261 [1:01:36<15:52,  3.64s/it]

    2.59 Seconds to search and locate 98 windows...
    2 cars found


     79%|███████▉  | 1000/1261 [1:01:40<15:51,  3.64s/it]

    2.66 Seconds to search and locate 104 windows...
    2 cars found


     79%|███████▉  | 1001/1261 [1:01:43<15:54,  3.67s/it]

    2.75 Seconds to search and locate 102 windows...
    2 cars found


     79%|███████▉  | 1002/1261 [1:01:47<15:44,  3.65s/it]

    2.59 Seconds to search and locate 109 windows...
    2 cars found


     80%|███████▉  | 1003/1261 [1:01:51<15:42,  3.65s/it]

    2.62 Seconds to search and locate 112 windows...
    2 cars found


     80%|███████▉  | 1004/1261 [1:01:54<15:41,  3.66s/it]

    2.7 Seconds to search and locate 108 windows...
    2 cars found


     80%|███████▉  | 1005/1261 [1:01:58<15:38,  3.66s/it]

    2.68 Seconds to search and locate 114 windows...
    2 cars found


     80%|███████▉  | 1006/1261 [1:02:02<15:32,  3.66s/it]

    2.64 Seconds to search and locate 112 windows...
    2 cars found


     80%|███████▉  | 1007/1261 [1:02:05<15:28,  3.66s/it]

    2.59 Seconds to search and locate 106 windows...
    2 cars found


     80%|███████▉  | 1008/1261 [1:02:09<15:27,  3.66s/it]

    2.71 Seconds to search and locate 93 windows...
    2 cars found


     80%|████████  | 1009/1261 [1:02:13<15:28,  3.68s/it]

    2.72 Seconds to search and locate 97 windows...
    2 cars found


     80%|████████  | 1010/1261 [1:02:16<15:17,  3.66s/it]

    2.6 Seconds to search and locate 89 windows...
    2 cars found


     80%|████████  | 1011/1261 [1:02:20<15:19,  3.68s/it]

    2.69 Seconds to search and locate 91 windows...
    2 cars found


     80%|████████  | 1012/1261 [1:02:24<15:14,  3.67s/it]

    2.69 Seconds to search and locate 94 windows...
    2 cars found


     80%|████████  | 1013/1261 [1:02:27<15:08,  3.66s/it]

    2.66 Seconds to search and locate 89 windows...
    2 cars found


     80%|████████  | 1014/1261 [1:02:31<15:03,  3.66s/it]

    2.62 Seconds to search and locate 88 windows...
    2 cars found


     80%|████████  | 1015/1261 [1:02:35<15:02,  3.67s/it]

    2.69 Seconds to search and locate 83 windows...
    2 cars found


     81%|████████  | 1016/1261 [1:02:38<14:58,  3.67s/it]

    2.68 Seconds to search and locate 89 windows...
    2 cars found


     81%|████████  | 1017/1261 [1:02:42<14:53,  3.66s/it]

    2.64 Seconds to search and locate 94 windows...
    2 cars found


     81%|████████  | 1018/1261 [1:02:46<14:53,  3.68s/it]

    2.64 Seconds to search and locate 80 windows...
    2 cars found


     81%|████████  | 1019/1261 [1:02:50<14:54,  3.70s/it]

    2.75 Seconds to search and locate 87 windows...
    2 cars found


     81%|████████  | 1020/1261 [1:02:53<14:51,  3.70s/it]

    2.72 Seconds to search and locate 90 windows...
    2 cars found


     81%|████████  | 1021/1261 [1:02:57<14:39,  3.67s/it]

    2.59 Seconds to search and locate 64 windows...
    2 cars found


     81%|████████  | 1022/1261 [1:03:01<14:39,  3.68s/it]

    2.63 Seconds to search and locate 65 windows...
    2 cars found


     81%|████████  | 1023/1261 [1:03:04<14:40,  3.70s/it]

    2.74 Seconds to search and locate 68 windows...
    2 cars found


     81%|████████  | 1024/1261 [1:03:08<14:35,  3.70s/it]

    2.69 Seconds to search and locate 65 windows...
    2 cars found


     81%|████████▏ | 1025/1261 [1:03:12<14:26,  3.67s/it]

    2.61 Seconds to search and locate 65 windows...
    2 cars found


     81%|████████▏ | 1026/1261 [1:03:15<14:22,  3.67s/it]

    2.6 Seconds to search and locate 72 windows...
    2 cars found


     81%|████████▏ | 1027/1261 [1:03:19<14:20,  3.68s/it]

    2.67 Seconds to search and locate 74 windows...
    2 cars found


     82%|████████▏ | 1028/1261 [1:03:23<14:18,  3.69s/it]

    2.68 Seconds to search and locate 72 windows...
    2 cars found


     82%|████████▏ | 1029/1261 [1:03:26<14:08,  3.66s/it]

    2.59 Seconds to search and locate 74 windows...
    2 cars found


     82%|████████▏ | 1030/1261 [1:03:30<14:07,  3.67s/it]

    2.65 Seconds to search and locate 74 windows...
    3 cars found


     82%|████████▏ | 1031/1261 [1:03:34<14:07,  3.68s/it]

    2.69 Seconds to search and locate 73 windows...
    2 cars found


     82%|████████▏ | 1032/1261 [1:03:37<14:00,  3.67s/it]

    2.62 Seconds to search and locate 76 windows...
    2 cars found


     82%|████████▏ | 1033/1261 [1:03:41<13:57,  3.67s/it]

    2.63 Seconds to search and locate 74 windows...
    2 cars found


     82%|████████▏ | 1034/1261 [1:03:45<13:56,  3.68s/it]

    2.68 Seconds to search and locate 83 windows...
    3 cars found


     82%|████████▏ | 1035/1261 [1:03:48<13:56,  3.70s/it]

    2.71 Seconds to search and locate 76 windows...
    2 cars found


     82%|████████▏ | 1036/1261 [1:03:52<13:49,  3.69s/it]

    2.62 Seconds to search and locate 74 windows...
    2 cars found


     82%|████████▏ | 1037/1261 [1:03:56<13:45,  3.69s/it]

    2.59 Seconds to search and locate 75 windows...
    2 cars found


     82%|████████▏ | 1038/1261 [1:03:59<13:44,  3.70s/it]

    2.7 Seconds to search and locate 78 windows...
    2 cars found


     82%|████████▏ | 1039/1261 [1:04:03<13:41,  3.70s/it]

    2.68 Seconds to search and locate 75 windows...
    2 cars found


     82%|████████▏ | 1040/1261 [1:04:07<13:29,  3.66s/it]

    2.56 Seconds to search and locate 76 windows...
    2 cars found


     83%|████████▎ | 1041/1261 [1:04:10<13:26,  3.67s/it]

    2.6 Seconds to search and locate 75 windows...
    2 cars found


     83%|████████▎ | 1042/1261 [1:04:14<13:26,  3.68s/it]

    2.68 Seconds to search and locate 78 windows...
    2 cars found


     83%|████████▎ | 1043/1261 [1:04:18<13:21,  3.68s/it]

    2.64 Seconds to search and locate 78 windows...
    2 cars found


     83%|████████▎ | 1044/1261 [1:04:21<13:14,  3.66s/it]

    2.61 Seconds to search and locate 74 windows...
    2 cars found


     83%|████████▎ | 1045/1261 [1:04:25<13:13,  3.67s/it]

    2.63 Seconds to search and locate 78 windows...
    2 cars found


     83%|████████▎ | 1046/1261 [1:04:29<13:11,  3.68s/it]

    2.69 Seconds to search and locate 80 windows...
    2 cars found


     83%|████████▎ | 1047/1261 [1:04:33<13:08,  3.69s/it]

    2.67 Seconds to search and locate 71 windows...
    2 cars found


     83%|████████▎ | 1048/1261 [1:04:36<12:57,  3.65s/it]

    2.55 Seconds to search and locate 71 windows...
    2 cars found


     83%|████████▎ | 1049/1261 [1:04:40<12:55,  3.66s/it]

    2.68 Seconds to search and locate 72 windows...
    2 cars found


     83%|████████▎ | 1050/1261 [1:04:44<12:54,  3.67s/it]

    2.69 Seconds to search and locate 73 windows...
    2 cars found


     83%|████████▎ | 1051/1261 [1:04:47<12:44,  3.64s/it]

    2.55 Seconds to search and locate 66 windows...
    2 cars found


     83%|████████▎ | 1052/1261 [1:04:51<12:42,  3.65s/it]

    2.58 Seconds to search and locate 70 windows...
    2 cars found


     84%|████████▎ | 1053/1261 [1:04:54<12:39,  3.65s/it]

    2.66 Seconds to search and locate 76 windows...
    2 cars found


     84%|████████▎ | 1054/1261 [1:04:58<12:38,  3.66s/it]

    2.7 Seconds to search and locate 68 windows...
    2 cars found


     84%|████████▎ | 1055/1261 [1:05:02<12:32,  3.65s/it]

    2.6 Seconds to search and locate 62 windows...
    2 cars found


     84%|████████▎ | 1056/1261 [1:05:05<12:29,  3.65s/it]

    2.57 Seconds to search and locate 61 windows...
    2 cars found


     84%|████████▍ | 1057/1261 [1:05:09<12:31,  3.68s/it]

    2.75 Seconds to search and locate 64 windows...
    2 cars found


     84%|████████▍ | 1058/1261 [1:05:13<12:32,  3.71s/it]

    2.71 Seconds to search and locate 74 windows...
    2 cars found


     84%|████████▍ | 1059/1261 [1:05:17<12:27,  3.70s/it]

    2.67 Seconds to search and locate 79 windows...
    2 cars found


     84%|████████▍ | 1060/1261 [1:05:20<12:28,  3.72s/it]

    2.69 Seconds to search and locate 78 windows...
    2 cars found


     84%|████████▍ | 1061/1261 [1:05:24<12:27,  3.74s/it]

    2.72 Seconds to search and locate 74 windows...
    2 cars found


     84%|████████▍ | 1062/1261 [1:05:28<12:26,  3.75s/it]

    2.73 Seconds to search and locate 71 windows...
    2 cars found


     84%|████████▍ | 1063/1261 [1:05:32<12:17,  3.72s/it]

    2.66 Seconds to search and locate 69 windows...
    2 cars found


     84%|████████▍ | 1064/1261 [1:05:35<12:13,  3.72s/it]

    2.66 Seconds to search and locate 70 windows...
    2 cars found


     84%|████████▍ | 1065/1261 [1:05:39<12:11,  3.73s/it]

    2.74 Seconds to search and locate 71 windows...
    2 cars found


     85%|████████▍ | 1066/1261 [1:05:43<12:08,  3.74s/it]

    2.72 Seconds to search and locate 73 windows...
    2 cars found


     85%|████████▍ | 1067/1261 [1:05:47<12:04,  3.73s/it]

    2.66 Seconds to search and locate 72 windows...
    2 cars found


     85%|████████▍ | 1068/1261 [1:05:50<12:00,  3.73s/it]

    2.67 Seconds to search and locate 77 windows...
    2 cars found


     85%|████████▍ | 1069/1261 [1:05:54<11:54,  3.72s/it]

    2.68 Seconds to search and locate 82 windows...
    2 cars found


     85%|████████▍ | 1070/1261 [1:05:58<11:54,  3.74s/it]

    2.74 Seconds to search and locate 85 windows...
    2 cars found


     85%|████████▍ | 1071/1261 [1:06:01<11:47,  3.73s/it]

    2.66 Seconds to search and locate 73 windows...
    2 cars found


     85%|████████▌ | 1072/1261 [1:06:05<11:44,  3.73s/it]

    2.67 Seconds to search and locate 71 windows...
    2 cars found


     85%|████████▌ | 1073/1261 [1:06:09<11:44,  3.75s/it]

    2.75 Seconds to search and locate 70 windows...
    2 cars found


     85%|████████▌ | 1074/1261 [1:06:13<11:39,  3.74s/it]

    2.69 Seconds to search and locate 68 windows...
    2 cars found


     85%|████████▌ | 1075/1261 [1:06:16<11:27,  3.70s/it]

    2.57 Seconds to search and locate 72 windows...
    2 cars found


     85%|████████▌ | 1076/1261 [1:06:20<11:22,  3.69s/it]

    2.65 Seconds to search and locate 80 windows...
    2 cars found


     85%|████████▌ | 1077/1261 [1:06:24<11:20,  3.70s/it]

    2.7 Seconds to search and locate 77 windows...
    2 cars found


     85%|████████▌ | 1078/1261 [1:06:27<11:14,  3.68s/it]

    2.63 Seconds to search and locate 77 windows...
    2 cars found


     86%|████████▌ | 1079/1261 [1:06:31<11:07,  3.67s/it]

    2.59 Seconds to search and locate 78 windows...
    2 cars found


     86%|████████▌ | 1080/1261 [1:06:35<11:05,  3.68s/it]

    2.69 Seconds to search and locate 74 windows...
    2 cars found


     86%|████████▌ | 1081/1261 [1:06:38<10:59,  3.66s/it]

    2.62 Seconds to search and locate 61 windows...
    2 cars found


     86%|████████▌ | 1082/1261 [1:06:42<10:55,  3.66s/it]

    2.63 Seconds to search and locate 60 windows...
    2 cars found


     86%|████████▌ | 1083/1261 [1:06:46<10:50,  3.65s/it]

    2.56 Seconds to search and locate 62 windows...
    2 cars found


     86%|████████▌ | 1084/1261 [1:06:49<10:50,  3.67s/it]

    2.69 Seconds to search and locate 56 windows...
    2 cars found


     86%|████████▌ | 1085/1261 [1:06:53<10:47,  3.68s/it]

    2.68 Seconds to search and locate 60 windows...
    2 cars found


     86%|████████▌ | 1086/1261 [1:06:57<10:40,  3.66s/it]

    2.59 Seconds to search and locate 64 windows...
    2 cars found


     86%|████████▌ | 1087/1261 [1:07:00<10:36,  3.66s/it]

    2.59 Seconds to search and locate 70 windows...
    2 cars found


     86%|████████▋ | 1088/1261 [1:07:04<10:34,  3.67s/it]

    2.69 Seconds to search and locate 69 windows...
    2 cars found


     86%|████████▋ | 1089/1261 [1:07:08<10:29,  3.66s/it]

    2.63 Seconds to search and locate 58 windows...
    2 cars found


     86%|████████▋ | 1090/1261 [1:07:11<10:26,  3.66s/it]

    2.63 Seconds to search and locate 68 windows...
    2 cars found


     87%|████████▋ | 1091/1261 [1:07:15<10:25,  3.68s/it]

    2.66 Seconds to search and locate 69 windows...
    2 cars found


     87%|████████▋ | 1092/1261 [1:07:19<10:20,  3.67s/it]

    2.67 Seconds to search and locate 67 windows...
    2 cars found


     87%|████████▋ | 1093/1261 [1:07:22<10:14,  3.66s/it]

    2.61 Seconds to search and locate 59 windows...
    2 cars found


     87%|████████▋ | 1094/1261 [1:07:26<10:12,  3.67s/it]

    2.61 Seconds to search and locate 68 windows...
    2 cars found


     87%|████████▋ | 1095/1261 [1:07:30<10:09,  3.67s/it]

    2.69 Seconds to search and locate 69 windows...
    2 cars found


     87%|████████▋ | 1096/1261 [1:07:33<10:06,  3.68s/it]

    2.69 Seconds to search and locate 64 windows...
    2 cars found


     87%|████████▋ | 1097/1261 [1:07:37<09:57,  3.64s/it]

    2.58 Seconds to search and locate 65 windows...
    2 cars found


     87%|████████▋ | 1098/1261 [1:07:41<09:55,  3.65s/it]

    2.61 Seconds to search and locate 61 windows...
    2 cars found


     87%|████████▋ | 1099/1261 [1:07:44<09:54,  3.67s/it]

    2.71 Seconds to search and locate 66 windows...
    2 cars found


     87%|████████▋ | 1100/1261 [1:07:48<09:51,  3.67s/it]

    2.66 Seconds to search and locate 60 windows...
    2 cars found


     87%|████████▋ | 1101/1261 [1:07:51<09:42,  3.64s/it]

    2.57 Seconds to search and locate 63 windows...
    2 cars found


     87%|████████▋ | 1102/1261 [1:07:55<09:40,  3.65s/it]

    2.61 Seconds to search and locate 64 windows...
    2 cars found


     87%|████████▋ | 1103/1261 [1:07:59<09:38,  3.66s/it]

    2.69 Seconds to search and locate 60 windows...
    2 cars found


     88%|████████▊ | 1104/1261 [1:08:03<09:34,  3.66s/it]

    2.64 Seconds to search and locate 60 windows...
    2 cars found


     88%|████████▊ | 1105/1261 [1:08:06<09:29,  3.65s/it]

    2.59 Seconds to search and locate 53 windows...
    2 cars found


     88%|████████▊ | 1106/1261 [1:08:10<09:25,  3.65s/it]

    2.66 Seconds to search and locate 56 windows...
    2 cars found


     88%|████████▊ | 1107/1261 [1:08:14<09:26,  3.68s/it]

    2.71 Seconds to search and locate 50 windows...
    2 cars found


     88%|████████▊ | 1108/1261 [1:08:17<09:19,  3.65s/it]

    2.59 Seconds to search and locate 47 windows...
    2 cars found


     88%|████████▊ | 1109/1261 [1:08:21<09:14,  3.64s/it]

    2.56 Seconds to search and locate 45 windows...
    2 cars found


     88%|████████▊ | 1110/1261 [1:08:24<09:12,  3.66s/it]

    2.67 Seconds to search and locate 51 windows...
    2 cars found


     88%|████████▊ | 1111/1261 [1:08:28<09:09,  3.67s/it]

    2.68 Seconds to search and locate 43 windows...
    1 cars found


     88%|████████▊ | 1112/1261 [1:08:32<09:02,  3.64s/it]

    2.59 Seconds to search and locate 42 windows...
    2 cars found


     88%|████████▊ | 1113/1261 [1:08:35<09:01,  3.66s/it]

    2.63 Seconds to search and locate 47 windows...
    2 cars found


     88%|████████▊ | 1114/1261 [1:08:39<08:56,  3.65s/it]

    2.64 Seconds to search and locate 50 windows...
    2 cars found


     88%|████████▊ | 1115/1261 [1:08:43<08:56,  3.68s/it]

    2.73 Seconds to search and locate 49 windows...
    1 cars found


     89%|████████▊ | 1116/1261 [1:08:46<08:49,  3.65s/it]

    2.58 Seconds to search and locate 50 windows...
    2 cars found


     89%|████████▊ | 1117/1261 [1:08:50<08:44,  3.64s/it]

    2.6 Seconds to search and locate 48 windows...
    2 cars found


     89%|████████▊ | 1118/1261 [1:08:54<08:43,  3.66s/it]

    2.7 Seconds to search and locate 44 windows...
    2 cars found


     89%|████████▊ | 1119/1261 [1:08:57<08:41,  3.67s/it]

    2.68 Seconds to search and locate 48 windows...
    2 cars found


     89%|████████▉ | 1120/1261 [1:09:01<08:35,  3.66s/it]

    2.59 Seconds to search and locate 50 windows...
    2 cars found


     89%|████████▉ | 1121/1261 [1:09:05<08:33,  3.67s/it]

    2.69 Seconds to search and locate 50 windows...
    1 cars found


     89%|████████▉ | 1122/1261 [1:09:08<08:31,  3.68s/it]

    2.69 Seconds to search and locate 55 windows...
    2 cars found


     89%|████████▉ | 1123/1261 [1:09:12<08:25,  3.66s/it]

    2.65 Seconds to search and locate 59 windows...
    2 cars found


     89%|████████▉ | 1124/1261 [1:09:16<08:21,  3.66s/it]

    2.58 Seconds to search and locate 61 windows...
    3 cars found


     89%|████████▉ | 1125/1261 [1:09:19<08:17,  3.65s/it]

    2.66 Seconds to search and locate 59 windows...
    2 cars found


     89%|████████▉ | 1126/1261 [1:09:23<08:14,  3.67s/it]

    2.69 Seconds to search and locate 57 windows...
    2 cars found


     89%|████████▉ | 1127/1261 [1:09:27<08:07,  3.64s/it]

    2.57 Seconds to search and locate 47 windows...
    2 cars found


     89%|████████▉ | 1128/1261 [1:09:30<08:04,  3.64s/it]

    2.6 Seconds to search and locate 48 windows...
    2 cars found


     90%|████████▉ | 1129/1261 [1:09:34<08:02,  3.65s/it]

    2.69 Seconds to search and locate 46 windows...
    1 cars found


     90%|████████▉ | 1130/1261 [1:09:38<07:57,  3.64s/it]

    2.62 Seconds to search and locate 40 windows...
    1 cars found


     90%|████████▉ | 1131/1261 [1:09:41<07:52,  3.64s/it]

    2.6 Seconds to search and locate 47 windows...
    2 cars found


     90%|████████▉ | 1132/1261 [1:09:45<07:49,  3.64s/it]

    2.64 Seconds to search and locate 53 windows...
    2 cars found


     90%|████████▉ | 1133/1261 [1:09:49<07:47,  3.65s/it]

    2.68 Seconds to search and locate 52 windows...
    2 cars found


     90%|████████▉ | 1134/1261 [1:09:52<07:40,  3.63s/it]

    2.59 Seconds to search and locate 51 windows...
    2 cars found


     90%|█████████ | 1135/1261 [1:09:56<07:38,  3.64s/it]

    2.58 Seconds to search and locate 46 windows...
    2 cars found


     90%|█████████ | 1136/1261 [1:09:59<07:36,  3.66s/it]

    2.7 Seconds to search and locate 50 windows...
    3 cars found


     90%|█████████ | 1137/1261 [1:10:03<07:34,  3.67s/it]

    2.69 Seconds to search and locate 57 windows...
    3 cars found


     90%|█████████ | 1138/1261 [1:10:07<07:27,  3.63s/it]

    2.57 Seconds to search and locate 46 windows...
    2 cars found


     90%|█████████ | 1139/1261 [1:10:10<07:24,  3.64s/it]

    2.6 Seconds to search and locate 41 windows...
    1 cars found


     90%|█████████ | 1140/1261 [1:10:14<07:23,  3.66s/it]

    2.68 Seconds to search and locate 46 windows...
    2 cars found


     90%|█████████ | 1141/1261 [1:10:18<07:18,  3.65s/it]

    2.62 Seconds to search and locate 40 windows...
    2 cars found


     91%|█████████ | 1142/1261 [1:10:21<07:11,  3.63s/it]

    2.56 Seconds to search and locate 42 windows...
    2 cars found


     91%|█████████ | 1143/1261 [1:10:25<07:09,  3.64s/it]

    2.65 Seconds to search and locate 45 windows...
    2 cars found


     91%|█████████ | 1144/1261 [1:10:29<07:07,  3.65s/it]

    2.67 Seconds to search and locate 42 windows...
    2 cars found


     91%|█████████ | 1145/1261 [1:10:32<07:01,  3.64s/it]

    2.61 Seconds to search and locate 41 windows...
    2 cars found


     91%|█████████ | 1146/1261 [1:10:36<06:58,  3.64s/it]

    2.58 Seconds to search and locate 44 windows...
    2 cars found


     91%|█████████ | 1147/1261 [1:10:39<06:54,  3.64s/it]

    2.63 Seconds to search and locate 41 windows...
    2 cars found


     91%|█████████ | 1148/1261 [1:10:43<06:55,  3.68s/it]

    2.77 Seconds to search and locate 35 windows...
    1 cars found


     91%|█████████ | 1149/1261 [1:10:47<06:48,  3.65s/it]

    2.58 Seconds to search and locate 42 windows...
    1 cars found


     91%|█████████ | 1150/1261 [1:10:51<06:45,  3.66s/it]

    2.61 Seconds to search and locate 39 windows...
    1 cars found


     91%|█████████▏| 1151/1261 [1:10:54<06:42,  3.66s/it]

    2.68 Seconds to search and locate 39 windows...
    1 cars found


     91%|█████████▏| 1152/1261 [1:10:58<06:38,  3.66s/it]

    2.65 Seconds to search and locate 41 windows...
    1 cars found


     91%|█████████▏| 1153/1261 [1:11:01<06:32,  3.64s/it]

    2.58 Seconds to search and locate 42 windows...
    1 cars found


     92%|█████████▏| 1154/1261 [1:11:05<06:30,  3.65s/it]

    2.65 Seconds to search and locate 37 windows...
    1 cars found


     92%|█████████▏| 1155/1261 [1:11:09<06:27,  3.65s/it]

    2.67 Seconds to search and locate 44 windows...
    1 cars found


     92%|█████████▏| 1156/1261 [1:11:12<06:24,  3.66s/it]

    2.69 Seconds to search and locate 39 windows...
    1 cars found


     92%|█████████▏| 1157/1261 [1:11:16<06:19,  3.65s/it]

    2.58 Seconds to search and locate 38 windows...
    1 cars found


     92%|█████████▏| 1158/1261 [1:11:20<06:16,  3.65s/it]

    2.66 Seconds to search and locate 36 windows...
    1 cars found


     92%|█████████▏| 1159/1261 [1:11:23<06:13,  3.67s/it]

    2.7 Seconds to search and locate 37 windows...
    1 cars found


     92%|█████████▏| 1160/1261 [1:11:27<06:08,  3.65s/it]

    2.6 Seconds to search and locate 42 windows...
    1 cars found


     92%|█████████▏| 1161/1261 [1:11:31<06:04,  3.65s/it]

    2.58 Seconds to search and locate 39 windows...
    1 cars found


     92%|█████████▏| 1162/1261 [1:11:34<06:02,  3.67s/it]

    2.7 Seconds to search and locate 38 windows...
    1 cars found


     92%|█████████▏| 1163/1261 [1:11:38<05:59,  3.67s/it]

    2.67 Seconds to search and locate 44 windows...
    1 cars found


     92%|█████████▏| 1164/1261 [1:11:42<05:54,  3.66s/it]

    2.64 Seconds to search and locate 40 windows...
    1 cars found


     92%|█████████▏| 1165/1261 [1:11:45<05:52,  3.67s/it]

    2.64 Seconds to search and locate 44 windows...
    1 cars found


     92%|█████████▏| 1166/1261 [1:11:49<05:49,  3.68s/it]

    2.68 Seconds to search and locate 41 windows...
    1 cars found


     93%|█████████▎| 1167/1261 [1:11:53<05:45,  3.68s/it]

    2.68 Seconds to search and locate 43 windows...
    1 cars found


     93%|█████████▎| 1168/1261 [1:11:56<05:40,  3.66s/it]

    2.57 Seconds to search and locate 40 windows...
    1 cars found


     93%|█████████▎| 1169/1261 [1:12:00<05:36,  3.66s/it]

    2.65 Seconds to search and locate 39 windows...
    1 cars found


     93%|█████████▎| 1170/1261 [1:12:04<05:33,  3.66s/it]

    2.67 Seconds to search and locate 39 windows...
    1 cars found


     93%|█████████▎| 1171/1261 [1:12:07<05:27,  3.64s/it]

    2.6 Seconds to search and locate 35 windows...
    1 cars found


     93%|█████████▎| 1172/1261 [1:12:11<05:24,  3.65s/it]

    2.61 Seconds to search and locate 33 windows...
    1 cars found


     93%|█████████▎| 1173/1261 [1:12:15<05:22,  3.67s/it]

    2.69 Seconds to search and locate 36 windows...
    1 cars found


     93%|█████████▎| 1174/1261 [1:12:18<05:18,  3.66s/it]

    2.67 Seconds to search and locate 30 windows...
    1 cars found


     93%|█████████▎| 1175/1261 [1:12:22<05:12,  3.64s/it]

    2.59 Seconds to search and locate 33 windows...
    1 cars found


     93%|█████████▎| 1176/1261 [1:12:26<05:10,  3.65s/it]

    2.61 Seconds to search and locate 36 windows...
    1 cars found


     93%|█████████▎| 1177/1261 [1:12:29<05:06,  3.65s/it]

    2.68 Seconds to search and locate 36 windows...
    1 cars found


     93%|█████████▎| 1178/1261 [1:12:33<05:03,  3.66s/it]

    2.68 Seconds to search and locate 33 windows...
    1 cars found


     93%|█████████▎| 1179/1261 [1:12:36<04:57,  3.63s/it]

    2.55 Seconds to search and locate 33 windows...
    1 cars found


     94%|█████████▎| 1180/1261 [1:12:40<04:56,  3.66s/it]

    2.7 Seconds to search and locate 31 windows...
    1 cars found


     94%|█████████▎| 1181/1261 [1:12:44<04:54,  3.68s/it]

    2.7 Seconds to search and locate 32 windows...
    1 cars found


     94%|█████████▎| 1182/1261 [1:12:48<04:49,  3.66s/it]

    2.62 Seconds to search and locate 33 windows...
    1 cars found


     94%|█████████▍| 1183/1261 [1:12:51<04:44,  3.65s/it]

    2.59 Seconds to search and locate 35 windows...
    1 cars found


     94%|█████████▍| 1184/1261 [1:12:55<04:41,  3.66s/it]

    2.67 Seconds to search and locate 39 windows...
    1 cars found


     94%|█████████▍| 1185/1261 [1:12:59<04:38,  3.66s/it]

    2.67 Seconds to search and locate 33 windows...
    1 cars found


     94%|█████████▍| 1186/1261 [1:13:02<04:33,  3.65s/it]

    2.62 Seconds to search and locate 35 windows...
    1 cars found


     94%|█████████▍| 1187/1261 [1:13:06<04:30,  3.65s/it]

    2.59 Seconds to search and locate 32 windows...
    1 cars found


     94%|█████████▍| 1188/1261 [1:13:09<04:26,  3.65s/it]

    2.66 Seconds to search and locate 31 windows...
    1 cars found


     94%|█████████▍| 1189/1261 [1:13:13<04:24,  3.67s/it]

    2.74 Seconds to search and locate 31 windows...
    1 cars found


     94%|█████████▍| 1190/1261 [1:13:17<04:19,  3.65s/it]

    2.58 Seconds to search and locate 32 windows...
    1 cars found


     94%|█████████▍| 1191/1261 [1:13:20<04:14,  3.64s/it]

    2.56 Seconds to search and locate 31 windows...
    1 cars found


     95%|█████████▍| 1192/1261 [1:13:24<04:12,  3.65s/it]

    2.68 Seconds to search and locate 32 windows...
    1 cars found


     95%|█████████▍| 1193/1261 [1:13:28<04:08,  3.66s/it]

    2.67 Seconds to search and locate 31 windows...
    1 cars found


     95%|█████████▍| 1194/1261 [1:13:31<04:04,  3.65s/it]

    2.6 Seconds to search and locate 37 windows...
    1 cars found


     95%|█████████▍| 1195/1261 [1:13:35<04:01,  3.66s/it]

    2.66 Seconds to search and locate 32 windows...
    1 cars found


     95%|█████████▍| 1196/1261 [1:13:39<03:57,  3.66s/it]

    2.66 Seconds to search and locate 30 windows...
    1 cars found


     95%|█████████▍| 1197/1261 [1:13:42<03:53,  3.65s/it]

    2.65 Seconds to search and locate 28 windows...
    1 cars found


     95%|█████████▌| 1198/1261 [1:13:46<03:50,  3.66s/it]

    2.63 Seconds to search and locate 30 windows...
    1 cars found


     95%|█████████▌| 1199/1261 [1:13:50<03:46,  3.66s/it]

    2.65 Seconds to search and locate 33 windows...
    1 cars found


     95%|█████████▌| 1200/1261 [1:13:53<03:43,  3.67s/it]

    2.7 Seconds to search and locate 34 windows...
    1 cars found


     95%|█████████▌| 1201/1261 [1:13:57<03:37,  3.63s/it]

    2.56 Seconds to search and locate 31 windows...
    1 cars found


     95%|█████████▌| 1202/1261 [1:14:01<03:34,  3.63s/it]

    2.55 Seconds to search and locate 32 windows...
    1 cars found


     95%|█████████▌| 1203/1261 [1:14:04<03:32,  3.66s/it]

    2.7 Seconds to search and locate 28 windows...
    1 cars found


     95%|█████████▌| 1204/1261 [1:14:08<03:28,  3.65s/it]

    2.63 Seconds to search and locate 34 windows...
    1 cars found


     96%|█████████▌| 1205/1261 [1:14:11<03:23,  3.63s/it]

    2.59 Seconds to search and locate 30 windows...
    1 cars found


     96%|█████████▌| 1206/1261 [1:14:15<03:20,  3.64s/it]

    2.63 Seconds to search and locate 30 windows...
    1 cars found


     96%|█████████▌| 1207/1261 [1:14:19<03:16,  3.64s/it]

    2.66 Seconds to search and locate 28 windows...
    1 cars found


     96%|█████████▌| 1208/1261 [1:14:22<03:12,  3.63s/it]

    2.63 Seconds to search and locate 30 windows...
    1 cars found


     96%|█████████▌| 1209/1261 [1:14:26<03:09,  3.64s/it]

    2.58 Seconds to search and locate 29 windows...
    1 cars found


     96%|█████████▌| 1210/1261 [1:14:30<03:05,  3.64s/it]

    2.66 Seconds to search and locate 32 windows...
    1 cars found


     96%|█████████▌| 1211/1261 [1:14:33<03:02,  3.66s/it]

    2.7 Seconds to search and locate 34 windows...
    1 cars found


     96%|█████████▌| 1212/1261 [1:14:37<02:58,  3.63s/it]

    2.58 Seconds to search and locate 34 windows...
    1 cars found


     96%|█████████▌| 1213/1261 [1:14:41<02:55,  3.66s/it]

    2.68 Seconds to search and locate 34 windows...
    1 cars found


     96%|█████████▋| 1214/1261 [1:14:44<02:52,  3.67s/it]

    2.68 Seconds to search and locate 41 windows...
    1 cars found


     96%|█████████▋| 1215/1261 [1:14:48<02:48,  3.67s/it]

    2.67 Seconds to search and locate 38 windows...
    1 cars found


     96%|█████████▋| 1216/1261 [1:14:52<02:44,  3.65s/it]

    2.6 Seconds to search and locate 41 windows...
    1 cars found


     97%|█████████▋| 1217/1261 [1:14:55<02:40,  3.65s/it]

    2.61 Seconds to search and locate 39 windows...
    1 cars found


     97%|█████████▋| 1218/1261 [1:14:59<02:37,  3.66s/it]

    2.7 Seconds to search and locate 42 windows...
    1 cars found


     97%|█████████▋| 1219/1261 [1:15:03<02:33,  3.66s/it]

    2.67 Seconds to search and locate 38 windows...
    1 cars found


     97%|█████████▋| 1220/1261 [1:15:06<02:29,  3.64s/it]

    2.57 Seconds to search and locate 37 windows...
    1 cars found


     97%|█████████▋| 1221/1261 [1:15:10<02:25,  3.64s/it]

    2.67 Seconds to search and locate 38 windows...
    1 cars found


     97%|█████████▋| 1222/1261 [1:15:14<02:23,  3.67s/it]

    2.73 Seconds to search and locate 40 windows...
    1 cars found


     97%|█████████▋| 1223/1261 [1:15:17<02:18,  3.66s/it]

    2.65 Seconds to search and locate 40 windows...
    1 cars found


     97%|█████████▋| 1224/1261 [1:15:21<02:15,  3.66s/it]

    2.6 Seconds to search and locate 37 windows...
    1 cars found


     97%|█████████▋| 1225/1261 [1:15:25<02:11,  3.66s/it]

    2.68 Seconds to search and locate 37 windows...
    1 cars found


     97%|█████████▋| 1226/1261 [1:15:28<02:08,  3.66s/it]

    2.67 Seconds to search and locate 42 windows...
    1 cars found


     97%|█████████▋| 1227/1261 [1:15:32<02:03,  3.63s/it]

    2.57 Seconds to search and locate 45 windows...
    1 cars found


     97%|█████████▋| 1228/1261 [1:15:35<02:00,  3.65s/it]

    2.63 Seconds to search and locate 42 windows...
    1 cars found


     97%|█████████▋| 1229/1261 [1:15:39<01:56,  3.65s/it]

    2.68 Seconds to search and locate 35 windows...
    1 cars found


     98%|█████████▊| 1230/1261 [1:15:43<01:53,  3.66s/it]

    2.69 Seconds to search and locate 31 windows...
    1 cars found


     98%|█████████▊| 1231/1261 [1:15:46<01:49,  3.64s/it]

    2.59 Seconds to search and locate 34 windows...
    1 cars found


     98%|█████████▊| 1232/1261 [1:15:50<01:45,  3.63s/it]

    2.6 Seconds to search and locate 35 windows...
    1 cars found


     98%|█████████▊| 1233/1261 [1:15:54<01:42,  3.65s/it]

    2.7 Seconds to search and locate 33 windows...
    1 cars found


     98%|█████████▊| 1234/1261 [1:15:57<01:38,  3.63s/it]

    2.6 Seconds to search and locate 28 windows...
    1 cars found


     98%|█████████▊| 1235/1261 [1:16:01<01:34,  3.63s/it]

    2.59 Seconds to search and locate 32 windows...
    1 cars found


     98%|█████████▊| 1236/1261 [1:16:05<01:31,  3.64s/it]

    2.67 Seconds to search and locate 31 windows...
    1 cars found


     98%|█████████▊| 1237/1261 [1:16:08<01:27,  3.66s/it]

    2.68 Seconds to search and locate 35 windows...
    1 cars found


     98%|█████████▊| 1238/1261 [1:16:12<01:23,  3.64s/it]

    2.6 Seconds to search and locate 32 windows...
    1 cars found


     98%|█████████▊| 1239/1261 [1:16:16<01:20,  3.65s/it]

    2.64 Seconds to search and locate 34 windows...
    1 cars found


     98%|█████████▊| 1240/1261 [1:16:19<01:16,  3.66s/it]

    2.68 Seconds to search and locate 34 windows...
    1 cars found


     98%|█████████▊| 1241/1261 [1:16:23<01:13,  3.66s/it]

    2.66 Seconds to search and locate 35 windows...
    1 cars found


     98%|█████████▊| 1242/1261 [1:16:26<01:09,  3.64s/it]

    2.58 Seconds to search and locate 34 windows...
    1 cars found


     99%|█████████▊| 1243/1261 [1:16:30<01:05,  3.64s/it]

    2.62 Seconds to search and locate 30 windows...
    1 cars found


     99%|█████████▊| 1244/1261 [1:16:34<01:02,  3.66s/it]

    2.71 Seconds to search and locate 24 windows...
    1 cars found


     99%|█████████▊| 1245/1261 [1:16:37<00:58,  3.64s/it]

    2.6 Seconds to search and locate 24 windows...
    1 cars found


     99%|█████████▉| 1246/1261 [1:16:41<00:54,  3.66s/it]

    2.66 Seconds to search and locate 25 windows...
    1 cars found


     99%|█████████▉| 1247/1261 [1:16:45<00:51,  3.68s/it]

    2.69 Seconds to search and locate 31 windows...
    1 cars found


     99%|█████████▉| 1248/1261 [1:16:49<00:47,  3.69s/it]

    2.69 Seconds to search and locate 31 windows...
    1 cars found


     99%|█████████▉| 1249/1261 [1:16:52<00:43,  3.66s/it]

    2.62 Seconds to search and locate 29 windows...
    1 cars found


     99%|█████████▉| 1250/1261 [1:16:56<00:40,  3.67s/it]

    2.61 Seconds to search and locate 32 windows...
    1 cars found


     99%|█████████▉| 1251/1261 [1:17:00<00:36,  3.67s/it]

    2.68 Seconds to search and locate 35 windows...
    1 cars found


     99%|█████████▉| 1252/1261 [1:17:03<00:33,  3.68s/it]

    2.7 Seconds to search and locate 36 windows...
    2 cars found


     99%|█████████▉| 1253/1261 [1:17:07<00:29,  3.64s/it]

    2.56 Seconds to search and locate 34 windows...
    2 cars found


     99%|█████████▉| 1254/1261 [1:17:10<00:25,  3.64s/it]

    2.59 Seconds to search and locate 37 windows...
    2 cars found


    100%|█████████▉| 1255/1261 [1:17:14<00:22,  3.67s/it]

    2.72 Seconds to search and locate 48 windows...
    2 cars found


    100%|█████████▉| 1256/1261 [1:17:18<00:18,  3.66s/it]

    2.63 Seconds to search and locate 61 windows...
    2 cars found


    100%|█████████▉| 1257/1261 [1:17:21<00:14,  3.65s/it]

    2.6 Seconds to search and locate 61 windows...
    2 cars found


    100%|█████████▉| 1258/1261 [1:17:25<00:10,  3.65s/it]

    2.63 Seconds to search and locate 77 windows...
    2 cars found


    100%|█████████▉| 1259/1261 [1:17:29<00:07,  3.66s/it]

    2.69 Seconds to search and locate 87 windows...
    2 cars found


    100%|█████████▉| 1260/1261 [1:17:32<00:03,  3.66s/it]

    2.65 Seconds to search and locate 81 windows...
    2 cars found


    


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project_video_final.mp4 
    


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

In the above cell, please find the **VehicleDetectionOp** class which handles all of the work necessary to detect and draw boxes around vehicles. More specifically, the entire algorithm, which leverages many methods provided to us during class, is encapsulated inside of **VehicleDetectionOp#perform**.

Here is a code snippet taken from **VehicleDetectionOp#perform** which creates a heatmap, thresholds it to identify vehicle positions, then identified number of detected vehicles along with their relative positioning using scipy.ndimage.measurements.label(), and finally drew boxes on top of the source image:

```python
# Weed out anomolies by excepting detections where at least 
# 5 windows were predicted to have a vehicle in it.
heat = np.zeros_like(image[:,:,0]).astype(np.float)
heatmap = add_heat(heat, hot_windows)
heatmap = apply_threshold(heatmap, 5)
self.heat = heatmap

# using `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap
labels = label(heatmap)
print(labels[1], 'cars found')

# constructing bounding boxes to cover the area of each blob detected
result = draw_labeled_bboxes(np.copy(image*255), labels)
```

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are seven frames and their corresponding heatmaps:


```python
images = glob.glob('../output_images/project_video/*_IN.jpg')
vehicle_detection_op.vis_detections = False
vehicle_detection_op.vis_labels = False
vehicle_detection_op.vis_heat = True
for img_path in images:
    result = pipeline.process_image(mpimg.imread(img_path))
```

    2.67 Seconds to search and locate 69 windows...
    2 cars found



![png](output_38_1.png)


    2.68 Seconds to search and locate 34 windows...
    1 cars found



![png](output_38_3.png)


    2.71 Seconds to search and locate 79 windows...
    2 cars found



![png](output_38_5.png)


    2.7 Seconds to search and locate 47 windows...
    1 cars found



![png](output_38_7.png)


    2.69 Seconds to search and locate 26 windows...
    1 cars found



![png](output_38_9.png)


    2.7 Seconds to search and locate 80 windows...
    1 cars found



![png](output_38_11.png)


    2.73 Seconds to search and locate 114 windows...
    1 cars found



![png](output_38_13.png)


    2.69 Seconds to search and locate 114 windows...
    2 cars found



![png](output_38_15.png)


    2.67 Seconds to search and locate 72 windows...
    1 cars found



![png](output_38_17.png)


    2.66 Seconds to search and locate 127 windows...
    2 cars found



![png](output_38_19.png)


    2.59 Seconds to search and locate 87 windows...
    1 cars found



![png](output_38_21.png)


    2.59 Seconds to search and locate 64 windows...
    1 cars found



![png](output_38_23.png)


    2.64 Seconds to search and locate 144 windows...
    1 cars found



![png](output_38_25.png)


    2.71 Seconds to search and locate 150 windows...
    1 cars found



![png](output_38_27.png)


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all seven frames:


```python
images = glob.glob('../output_images/project_video/*_IN.jpg')
vehicle_detection_op.vis_detections = False
vehicle_detection_op.vis_labels = True
vehicle_detection_op.vis_heat = False
for img_path in images:
    result = pipeline.process_image(mpimg.imread(img_path))
```

    2.68 Seconds to search and locate 69 windows...
    2 cars found



![png](output_40_1.png)


    2.74 Seconds to search and locate 34 windows...
    1 cars found



![png](output_40_3.png)


    2.68 Seconds to search and locate 79 windows...
    2 cars found



![png](output_40_5.png)


    2.62 Seconds to search and locate 47 windows...
    1 cars found



![png](output_40_7.png)


    2.58 Seconds to search and locate 26 windows...
    1 cars found



![png](output_40_9.png)


    2.6 Seconds to search and locate 80 windows...
    1 cars found



![png](output_40_11.png)


    2.65 Seconds to search and locate 114 windows...
    1 cars found



![png](output_40_13.png)


    2.72 Seconds to search and locate 114 windows...
    2 cars found



![png](output_40_15.png)


    2.73 Seconds to search and locate 72 windows...
    1 cars found



![png](output_40_17.png)


    2.7 Seconds to search and locate 127 windows...
    2 cars found



![png](output_40_19.png)


    2.67 Seconds to search and locate 87 windows...
    1 cars found



![png](output_40_21.png)


    2.6 Seconds to search and locate 64 windows...
    1 cars found



![png](output_40_23.png)


    2.58 Seconds to search and locate 144 windows...
    1 cars found



![png](output_40_25.png)


    2.61 Seconds to search and locate 150 windows...
    1 cars found



![png](output_40_27.png)


### Here the resulting bounding boxes are drawn onto the last frame in the series:


```python
images = glob.glob('../output_images/project_video/*_IN.jpg')
vehicle_detection_op.vis_detections = False
vehicle_detection_op.vis_labels = False
vehicle_detection_op.vis_heat = False
for img_path in images:
    result = pipeline.process_image(mpimg.imread(img_path))
    cv2.imwrite('../output_images/project_video/{}_FINAL.jpg'.format(os.path.basename(img_path).split('.')[0]), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    PlotImageOp(result*255, title="{} - FINAL".format(img_path), cmap=None).perform()
```

    2.6 Seconds to search and locate 70 windows...
    2 cars found



![png](output_42_1.png)


    2.63 Seconds to search and locate 33 windows...
    1 cars found



![png](output_42_3.png)


    2.63 Seconds to search and locate 90 windows...
    2 cars found



![png](output_42_5.png)


    2.67 Seconds to search and locate 45 windows...
    1 cars found



![png](output_42_7.png)


    2.84 Seconds to search and locate 26 windows...
    1 cars found



![png](output_42_9.png)


    2.64 Seconds to search and locate 82 windows...
    1 cars found



![png](output_42_11.png)


    2.92 Seconds to search and locate 115 windows...
    1 cars found



![png](output_42_13.png)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 


---

The greatest challenge I faced during this process was minimizing processing time for each detection operation. As it stands with my current pipeline settings, I ultimately langed on a overlap value 0.9 because it resulted in the most accurate number of predictions across all frames. There are still a few frames which didn't make my `5` window threshold so no bounding box was drawn over the vehicle and I did see one or two anomolies creep into the video.

To make it more robust, I would track detected objects and smooth out the window over time using clever moving average algoithms for a cleaner detection experience. I would also implemented a "Left Lane Window Sliding", "Center Lane Window Sliding" and "Right Lane Window Sliding" modes which I'd use to adjust the windows in which I search against based on the lane I am in. I would brake the viewing space up into various sections to filter against and I'd leverage concurrency whereever possible to minimize lag between frames, etc. because in a real-time situation, speed is absolutely critical.
