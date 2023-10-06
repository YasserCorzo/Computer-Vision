import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matchPics import matchPics
from opts import get_opts

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover_img = cv2.imread('../data/cv_cover.jpg')
angles = []
num_matches = []
for i in range(36):
	#Rotate Image
	angle = i * 10
	print(angle)
	rotated_cv_cover_img = scipy.ndimage.rotate(cv_cover_img, angle, reshape=False)

	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover_img, rotated_cv_cover_img, opts)

	#Update histogram
	angles.append(angle)
	num_matches.append(matches.shape[0])

	pass # comment out when code is ready


#Display histogram
plt.bar(angles, num_matches, linewidth=10.0)
plt.xlabel("Angle of rotation (degrees)")
plt.ylabel("Number of matches")
plt.show()