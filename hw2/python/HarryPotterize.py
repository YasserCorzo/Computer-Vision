import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import *
from planarH import *
from myHelperFunctions import *
from helper import plotMatches

#Write script for Q2.2.4
opts = get_opts()

# reads images 
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# scale hp_cover image
hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

# compute matched features 
matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)
#plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

# coordinates of matched features
x1_match, x2_match = findMatches(matches, locs1, locs2)

# compute best homography and inliers
bestH2to1, inliers = computeH_ransac(x1_match, x2_match, opts)

# warp hp_cover image
warped_hp_cover = cv2.warpPerspective(hp_cover, bestH2to1, (cv_desk.shape[1], cv_desk.shape[0]))

# save warped image
cv2.imwrite('warped.jpg', warped_hp_cover)

# compose this warped image with the desk image
composite_img = compositeH(bestH2to1, hp_cover, cv_desk)

# save composite images
cv2.imwrite('composite_img.jpg', composite_img)