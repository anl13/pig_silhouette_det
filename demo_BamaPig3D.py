import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import os 
import time 
from tqdm import tqdm  
import argparse 
import glob
import numpy as np 
import json 
import pickle 
from tqdm import tqdm 

from detectron2.config import get_cfg 
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from point_rend import add_pointrend_config
import matplotlib.pyplot as plt 

class VisDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode 

        self.parallel = parallel 
        self.predictor = DefaultPredictor(cfg) 

    def run_on_image(self, image):
        '''
            image: (H,W,C) BGR by opencv 
        '''
        vis_output = None 
        predictions = self.predictor(image)
        image = image[:,:,::-1] # bgr to rgb
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "instances" in predictions: 
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        # default="configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco_mask.yaml",
        default = "output_20210225/config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--data3d_path", 
        help = "The path of BamaPig3D dataset",
        default = ""
    )
    parser.add_argument("--write_dir",
        help="Directory to write box and mask json files. ",
        default=""
    )
    return parser


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def convert_mask(masks):
    newmask = []
    for contours in masks: 
        a_object = []
        for contour in contours: 
            a_part = [] 
            for k in range(contour.shape[0]):
                a_part.append(contour[k].tolist())
            a_object.append(a_part)
        newmask.append(a_object)
    return newmask

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args() 
    setup_logger(name = "fvcore")
    logger = setup_logger() 
    logger.info("arguments: " + str(args))

    cfg=setup_cfg(args) 
    demo = VisDemo(cfg) 

    camids = [0,1,2,5,6,7,8,9,10,11]
    output_dir = args.write_dir 
    if output_dir == "": 
        output_dir = args.data3d_path 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
    
    box_folder = output_dir + "/boxes_pr"
    mask_folder = output_dir + "/masks_pr"
    if not os.path.exists(box_folder):
        os.makedirs(box_folder)
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    for i in tqdm(range(0,1750)):
        box_json_file = box_folder + "/{:06d}.json".format(i)
        mask_json_file = mask_folder + "/{:06d}.json".format(i) 
        box_json = {}
        mask_json = {} 
        values = {}
        for cam in camids: 
            data = {}
            imagename = args.data3d_path + "/images/cam{}/{:06d}.jpg".format(cam, i)
            img = cv2.imread(imagename)
            predictions, vis_output = demo.run_on_image(img)
            # if i == 0:
            #     out_image = vis_output.get_image()[:,:,::-1]
            #     savefile = "cam{}_{:06d}.jpg".format(cam, i)
            #     cv2.imwrite(savefile, out_image)

            instances = predictions['instances'].to(torch.device("cpu"))
            masks = np.asarray(instances.pred_masks)
            boxes = instances.pred_boxes.tensor.numpy()
            mask_num = masks.shape[0]
            mask_list = [] 
            box_json.update({cam:boxes.tolist()})
            for k in range(mask_num):
                a = masks[k].astype(np.uint8) * 255
                contours, hierarchy = cv2.findContours(a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                mask_list.append(contours)
            output_mask = convert_mask(mask_list)
            mask_json.update({cam:output_mask})
        with open(box_json_file, 'w') as f: 
            json.dump(box_json,f) 
        with open(mask_json_file, 'w') as f: 
            json.dump(mask_json, f) 
