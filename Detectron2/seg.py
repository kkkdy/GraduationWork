# import some common libraries
import numpy as np
import cv2

import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', help='foo help', required=True)
parser.add_argument('--name', help='foo help', required=True)
args = parser.parse_args()

root_name = args.dataroot
name = args.name

# make folder
if not os.path.exists(os.path.join('segment_data', root_name)):
    os.makedirs(os.path.join('segment_data', root_name))

for filename in os.listdir(root_name):
    if '_fake' in filename:
        print(filename)
        first, second = os.path.splitext(filename)
        im = cv2.imread(os.path.join(root_name, filename))
        im = cv2.resize(im, dsize=(349, 640))
        filename.replace("_fake", "")

        im_point = np.zeros((640, 349, 4), np.uint8)
        first, _ = os.path.splitext(filename)
        w, h = np.loadtxt(os.path.join('coordinate', name, "size_" + first + ".txt"))
        coor_point = np.loadtxt(os.path.join('coordinate', name, "coor_" + first + ".txt"))

        # print(im.shape[0])
        b_channel, g_channel, r_channel = cv2.split(im)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # creating a dummy alpha channel image
        im_point = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        for i in range(640):
            for j in range(349):
                if coor_point[i][j] == 1:
                    im_point[i][j] = im_point[i][j]
                else:
                    im_point[i][j] = [0, 0, 0, 1]

        # cv2.imshow("", im_point)
        # cv2.waitKey(0)

        # to PNG
        img_BGRA = cv2.resize(im_point, dsize=(int(w), int(h)))
        cv2.imwrite(os.path.join('segment_data', root_name, 'seg_' + filename), img_BGRA)
