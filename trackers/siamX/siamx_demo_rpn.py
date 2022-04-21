#!/usr/bin/python

import argparse
import glob
import cv2
import numpy as np

from siamx_rpn.net import *
from siamx_rpn.siam_rpn_tracker import siam_rpn_init, siam_rpn_track
from siamx_rpn.siamx_rpn_utils import get_axis_aligned_bbox, cxy_wh_2_rect, load_net


parser = argparse.ArgumentParser(description='PyTorch SiameseX demo')

parser.add_argument('--model', metavar='model', default='SiamRPNPPRes50', type=str,
                    help='which model to use.')
args = parser.parse_args()

# load net
net = eval(args.model)()
load_net('./cp/{}.pth'.format(args.model), net)
net.eval().cuda()

# image and init box
image_files = sorted(glob.glob('./data/bag/*.jpg'))
init_rbox = [334.02, 128.36, 438.19, 188.78, 396.39, 260.83, 292.23, 200.41]
[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_files[0])  # HxWxC
state = siam_rpn_init(im, target_pos, target_sz, net, args.model)

# tracking and visualization
toc = 0
for f, image_file in enumerate(image_files):
    im = cv2.imread(image_file)
    # print(im.shape)
    tic = cv2.getTickCount()
    state = siam_rpn_track(state, im)  # track
    toc += cv2.getTickCount()-tic
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    # print(res)
    cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
    cv2.imshow('SiamRPN', im)
    cv2.waitKey(1)

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
