import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    
    # estimate noise
    sigma_est = skimage.restoration.estimate_sigma(image, channel_axis=-1, average_sigmas=True)
    
    # denoise image with estimated sigma
    gaussian_img = skimage.filters.gaussian(image, sigma=sigma_est)
    
    # greyscale image
    greyscale_img = skimage.color.rgb2gray(gaussian_img)
    
    # set threshold to create binary image
    thresh = skimage.filters.threshold_otsu(greyscale_img)
    bw = greyscale_img < thresh
    
    # morphology (closing)
    bw = skimage.morphology.closing(bw, skimage.morphology.square(3))
    
    # label connected groups of character pixels
    label_img = skimage.morphology.label(bw, connectivity=2)
    
    # find locations of the labels
    label_props = skimage.measure.regionprops(label_img)
    
    # find the mean area of the labeled regions
    areas = []
    for region in label_props:
        areas.append(region.area)
    mean_area = sum(areas) / len(areas)
    
    # set new threshold for mean (after testing mean needs to be smaller)
    thresh_mean = mean_area / 3
    
    # calculate upper left and lower left coords of regions, skip small regions
    for region in label_props:
        if region.area >= thresh_mean:
            bboxes.append(region.bbox)

    return bboxes, bw