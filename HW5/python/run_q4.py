import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# function for making cropped image into a square depending on height and width
def pad_to_square(M):
    (height, width) = M.shape
    if height == width:
        height_pad = 0
        width_pad = 0
    
    # more padding added to width
    elif height > width:
        height_pad = height // 10
        width_pad = ((height - width) // 2) + height_pad
    
    # more padding added to height
    elif height < width:
        width_pad = width // 10
        height_pad = ((width - height) // 2) + width_pad

    # set tuples of pad height and width
    pad_height = (height_pad, height_pad)
    pad_width = (width_pad, width_pad)
    return np.pad(M, (pad_height, pad_width), mode='constant', constant_values=(1, 1))

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(np.logical_not(bw).astype(float), cmap="Greys")
    #plt.imshow(bw, cmap="Greys")
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    #print(bboxes)
    #print(len(bboxes))
    #print(bboxes[0])
    #print(bboxes.shape)

    # calculate heights of all the boxes
    heights = []
    for bbox in bboxes:
        height = bbox[2] - bbox[0]
        heights.append(height)
    heights = np.array(heights)

    # calculate mean height of all boxes
    mean_height = np.mean(heights)
    #print("mean height:", mean_height)

    # calculate the center coords of each box (center_y, center_x)
    # additonal values stored: upper left and lower left coords
    # (center_y, center_x, [y1, x1, y2, x2])
    center_coords = []
    for bbox in bboxes:
        center = [(bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2, bbox]
        center_coords.append(center)
    #print(center_coords)

    # sort the center coords list by y coord
    center_coords = sorted(center_coords, key=lambda x: x[0])
    #print(center_coords)
    #print("num boxes:", len(center_coords))

    # store y coord of first box of row (will be used for thresholding when to create a new row)
    first_box_center_y = center_coords[0][0]
    
    # list to store all rows
    rows = []

    # list to store current row
    curr_row = []

    # add first box to current row
    curr_row.append(center_coords[0])

    for center_coord in center_coords[1:]:

        if center_coord[0] > first_box_center_y + mean_height:
            # sort the current row we're building by increasing x values
            curr_row = sorted(curr_row, key=lambda x: x[1])

            # add current row we've built
            rows.append(curr_row)
            #print("new row added")

            # create new row with current box added
            curr_row = []
            curr_row.append(center_coord)

            # update the y value of the first box of new row
            first_box_center_y = center_coord[0]
        else:
            #print("added to current row")
            curr_row.append(center_coord)

    # sort remaining row (if there is one) not added by increasing x values and add it to list of rows
    if len(curr_row) != 0:
        curr_row = sorted(curr_row, key=lambda x: x[1])
        rows.append(curr_row)
    #print(rows)
    '''
    print(rows)
    print("num rows:", len(rows))
    num_boxes = 0
    for row in rows:
        num_boxes += len(row)
    print("num boxes in rows list:", num_boxes)
    '''

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################

    flattened_boxes = []

    for row in rows:
        flattened_boxes_curr_row = []
        for box_data in row:
            y1, x1, y2, x2 = box_data[2]

            # crop the bounding box
            crop_img = bw[y1:y2+1, x1:x2+1]
            '''
            plt.figure()
            plt.imshow(crop_img, cmap="Greys")
            plt.show()
            '''
            # pad cropped image to be square
            square_crop_img = pad_to_square(crop_img)
            '''
            plt.figure()
            plt.imshow(square_crop_img, cmap="Greys")
            plt.show()
            '''

            # resize cropped image dim 32x32
            crop_img = skimage.transform.resize(square_crop_img, (32, 32))
            '''
            plt.figure()
            plt.imshow(crop_img_resize, cmap="Greys")
            plt.show()
            '''
            
            kernel = np.ones((3,3), np.uint8) 
            crop_img = skimage.morphology.erosion(crop_img, kernel)

            # transpose image
            crop_img = crop_img.T
            '''
            plt.figure()
            plt.imshow(crop_img, cmap="Greys")
            plt.show()
            '''
            # flatten image
            flatten_crop_img = crop_img.flatten()

            flattened_boxes_curr_row.append(flatten_crop_img)

        flattened_boxes.append(flattened_boxes_curr_row)


    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    classes_dict = {}

    for (class_num, c) in enumerate(letters):
        classes_dict[class_num] = c
    
    print(classes_dict)

    for x in flattened_boxes:
        # forward prop
        h1 = forward(x, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        res = ''
        pred_classes = np.argmax(probs, axis=1)
        #print(pred_classes)
        for pred_c in pred_classes:
            res += classes_dict[pred_c]
    
        print(res)
