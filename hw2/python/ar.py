import numpy as np
import cv2
#Import necessary functions
from loadVid import *
from planarH import *
from matchPics import *
from opts import get_opts


#Write script for Q3.1
opts = get_opts()

ar_source = loadVid('../data/ar_source.mov')
book = loadVid('../data/book.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')

frame_width = book.shape[2]
frame_height = book.shape[1]
frame_rate = 15
# number of frames from book and ar video are different so choose the minimum
total_frames = min(book.shape[0], ar_source.shape[0])
# frame size of video must match that of book video
result = cv2.VideoWriter("../result/ar.avi", cv2.VideoWriter_fourcc(*'MJPG'), frame_rate, (frame_width, frame_height))

for i in range(total_frames):
    book_frame = book[i]
    ar_frame = ar_source[i]
    # remove top and bottom black bars from ar video
    ar_frame = ar_frame[44:-44,:, :]
    # ar and cv cover have different aspect ratios so make ar frame have the same aspect ratio as cv cover
    ar_frame = cropToAspectRatio(ar_frame, cv_cover)
    ar_frame = cv2.resize(ar_frame, (cv_cover.shape[1], cv_cover.shape[0]))
    
    # calculate matched point pairs
    matches, locs1, locs2 = matchPics(book_frame, cv_cover, opts)
    
    # retrieve matched points from book video and of cv cover
    x1_match, x2_match = findMatches(matches, locs1, locs2)

    # calculate best homography between cv cover and book video
    bestH2to1, inliers = computeH_ransac(x1_match, x2_match, opts)  
    
    # create composite frame of ar video and book video
    composite_frame = compositeH(bestH2to1, ar_frame, book_frame)

    result.write(composite_frame)


result.release()