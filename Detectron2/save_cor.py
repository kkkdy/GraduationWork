# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
import sys;

sys.path.insert(1, "projects/PointRend")
import point_rend

import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', help='foo help', required=True)
args = parser.parse_args()

root_name = args.dataroot
# dataroot = os.p
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
mask_rcnn_predictor = DefaultPredictor(cfg)

cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"
predictor = DefaultPredictor(cfg)

# make folder
if not os.path.exists(os.path.join('coordinate', root_name)):
    os.makedirs(os.path.join('coordinate', root_name))

for filename in os.listdir(root_name):
    print(filename)
    im = cv2.imread(os.path.join(root_name, filename))
    # cv2.imshow("",im)
    # cv2.waitKey(0)

    w = im.shape[1]
    h = im.shape[0]
    im = cv2.resize(im, dsize=(349, 640))
    # print(w,h)

    mask_rcnn_outputs = mask_rcnn_predictor(im)

    outputs = predictor(im)
    # print(outputs)

    # Show and compare two predictions:
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    mask_rcnn_result = v.draw_instance_predictions(mask_rcnn_outputs["instances"].to("cpu")).get_image()
    # cv2.imshow(mask_rcnn_result)
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    #    cv2.imshow("", np.concatenate((point_rend_result, mask_rcnn_result), axis=0)[:, :, ::-1])
    #    cv2.waitKey(0)

    true_coor = [[True for col in range(w)] for row in range(h)]
    # pointrend coordinates
    if (len(outputs["instances"]) != 0):
        for i in range(len(outputs["instances"])):
            if outputs["instances"].pred_classes[i] == 17 or outputs["instances"].pred_classes[i] == 16:
                coor_point = outputs["instances"].pred_masks[0].cpu().detach().numpy()
                name, _ = os.path.splitext(filename)
                # print(coor_point[0])
                np.savetxt(os.path.join('coordinate', root_name, "coor_" + name + ".txt"), coor_point)
                np.savetxt(os.path.join('coordinate', root_name, "size_" + name + ".txt"), [w, h])
            else:
                np.savetxt(os.path.join('coordinate', root_name, "coor_" + name + ".txt"), true_coor)
                np.savetxt(os.path.join('coordinate', root_name, "size_" + name + ".txt"), [w, h])
    else:
        np.savetxt(os.path.join('coordinate', root_name, "coor_" + name + ".txt"), true_coor)
        np.savetxt(os.path.join('coordinate', root_name, "size_" + name + ".txt"), [w, h])

        #    torch.save(coor_point, 'file.pt')

