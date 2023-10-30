import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import LucasKanade

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

num_frames = seq.shape[2]

# keep track of overall rects (with or w/o template correction)
wcrt_seq_rects = []
wcrt_seq_rects.append(rect)

# load rects w/o template correction
seq_rects = np.load('girlseqrects.npy')

fig, ax = plt.subplots()

# keep track of first template
T1 = seq[:, :, 0]

p = np.zeros(2)
p_acc = np.zeros(2)

for frame in range(num_frames - 1):
    ax.axis('on')
    print("frame:", frame)
    It = seq[:, :, frame]
    It1 = seq[:, :, frame + 1]
    wcrt_rect = wcrt_seq_rects[frame]
    rect = seq_rects[frame]

    # calculate movement vector 
    p_n = LucasKanade(It, It1, wcrt_rect, threshold, num_iters, p)

    # update step 
    pstar_n = LucasKanade(T1, It1, wcrt_seq_rects[0], threshold, num_iters)

    # update p
    if np.linalg.norm(pstar_n - p_acc - p) <= template_threshold:
        p = pstar_n - p_acc 
        p_acc += p
        updated_rect = np.array([wcrt_rect[0] + p[0], wcrt_rect[1] + p[1], wcrt_rect[2] + p[0], wcrt_rect[3] + p[1]])
        wcrt_seq_rects.append(updated_rect)
    else:
        wcrt_seq_rects.append(wcrt_rect)
    
    ax.add_patch(patches.Rectangle((wcrt_rect[0], wcrt_rect[1]), wcrt_rect[2]-wcrt_rect[0]+1, wcrt_rect[3]-wcrt_rect[1]+1, edgecolor='red', fill=False))
    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, edgecolor='blue', fill=False))
    ax.axis('off')
    plt.imshow(It1, cmap='gray')

    # report tracking performance
    if frame != 0 and ((frame == 1) or (frame % 20 == 0)):
        img_name = f'wcrt_frame_girl_{frame}.jpg'
        plt.savefig(img_name)
    plt.pause(0.01)
    ax.clear()

np.save('girlseqrects-wcrt.npy', wcrt_seq_rects)