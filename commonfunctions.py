import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv,rgba2rgb

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median, threshold_otsu,threshold_sauvola
from skimage.filters.rank import otsu
from skimage.morphology import disk
from skimage.feature import canny,hog
from skimage.measure import label
from skimage.color import label2rgb

from skimage.draw import polygon
from skimage.measure import find_contours
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line, rotate, resize
import numpy as np
import matplotlib as mpl
import collections 

from matplotlib import cm
# import cv2
from skimage.exposure import histogram,rescale_intensity,is_low_contrast
from skimage.draw import rectangle, polygon
from skimage.measure import find_contours
from skimage.morphology import binary_erosion,binary_dilation,skeletonize,binary_closing,binary_opening
from skimage.feature import canny
from skimage.filters import gaussian, median

from skimage import data,exposure
from PIL import Image
import PIL
import collections 
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import find_peaks

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt
from scipy import ndimage
# Show the figures / plots inside the notebook
from skimage.feature import hog
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from sklearn import svm

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 

def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

    
    
    
    
    
    
    
    
   