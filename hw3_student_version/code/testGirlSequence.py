import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import LucasKanade

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

num_frames = seq.shape[2]
w_rect = rect[2] - rect[0]
h_rect = rect[3] - rect[1]

# loop over video frames
seq_rects = []
seq_rects.append(rect)
fig, ax = plt.subplots()

p = np.zeros(2)

for frame in range(num_frames - 1):
    ax.axis('on')
    print("frame:", frame)
    It = seq[:, :, frame]
    It1 = seq[:, :, frame + 1]
    rect = seq_rects[frame]

    # calculate movement vector 
    dp_x, dp_y = LucasKanade(It, It1, rect, threshold, num_iters, p)

    # update p for next iteration
    p = np.array([dp_x, dp_y])

    # create rect for next frame
    rect1 = np.array([rect[0] + dp_x, rect[1] + dp_y, rect[0] + dp_x + w_rect, rect[1] + dp_y + h_rect])
    seq_rects.append(rect1)

    # Plot rect on frame
    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, edgecolor='red', fill=False))
    ax.axis('off')
    plt.imshow(It1, cmap='gray')

    # report tracking performance
    if frame != 0 and ((frame == 1) or (frame % 20 == 0)):
        img_name = f'frame_girl_{frame}.jpg'
        plt.savefig(img_name)
    plt.pause(0.01)
    ax.clear()

np.save('girlseqrects.npy', seq_rects)