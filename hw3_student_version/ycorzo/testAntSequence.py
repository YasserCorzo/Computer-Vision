import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from SubtractDominantMotion import SubtractDominantMotion
import time

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')

num_frames = seq.shape[2]

fig, ax = plt.subplots()

t0 = time.time()
for frame in range(num_frames - 1):
    ax.axis('on')
    print("frame:", frame)
    It = seq[:, :, frame]
    It1 = seq[:, :, frame + 1]

    # calculating binary mask where pixels crossing threshold are noted as 1
    mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance)

    # calculate location where binary mask has a 1
    x, y = np.where(mask == 1) 
    locs = np.vstack((x, y)).T

    # create patches around these locations
    
    for coor in range(locs.shape[0]):
        x, y = locs[coor]
        ax.add_patch(patches.Rectangle((y - 2, x - 2), 4, 4, color='blue', fill=True))
    
    ax.axis('off')
    plt.imshow(It1, cmap='gray')

    # report tracking performance
    if ((frame + 1) % 30 == 0):
        img_name = f'frame_ant_{frame + 1}.jpg'
        plt.savefig(img_name)
    plt.pause(0.01)
    ax.clear()
t1 = time.time()
dt = t1 - t0
print("runtime of ant sequence:", dt)