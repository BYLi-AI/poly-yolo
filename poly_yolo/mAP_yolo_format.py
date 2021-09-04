import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def yolo_to_coco(pred_path: str, gt_path: str, classes_path: str) -> None:
    """
    Converts predictions and labels in yolo format into coco compatible json files that are then evaluated.
    :param pred_path: Path to the text file with predictions in yolo format.
    :param gt_path: Path to the text file with labels in yolo format.
    :param classes_path: Path to the text file with classes contained in labels/predictions. One per line.
    """
    print('beginning conversion from yolo format to coco json files ...')
    coco_pred = []
    clss = []
    with open(classes_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            clss.append({'id': i, 'name': lines[i].rstrip()})
    coco_gt = {'annotations': [],
               'images': [],
               # TODO:categories could be loaded from classes.txt
               'categories': clss}
    img_name_to_id = {}
    img_id = 0
    with open(pred_path, 'r') as f1:
        pred_lines = f1.readlines()
    with open(gt_path, 'r') as f2:
        gt_lines = f2.readlines()

    id = 0
    for gt_line in gt_lines:
        gt_line = gt_line.rstrip()
        gt_img_path = gt_line.split(' ')[0]
        gt_data = gt_line.split(' ')
        gt_vectors = gt_data[1:]

        for gt_vector in gt_vectors:
            gt_vector = str_list_to_float_list(gt_vector)
            annotation = {'image_id': img_id,
                 'id': id,
                 'iscrowd': 0,
                 # area does not matter - in evaluation we are looking for area: all
                 'area': 1,
                 'category_id': int(gt_vector[4]),
                 'bbox': [gt_vector[0], gt_vector[1], gt_vector[2] - gt_vector[0], gt_vector[3] - gt_vector[1]]}
            if len(gt_vector) > 5:
                annotation['segmentation'] = [gt_vector[5:]]
            coco_gt['annotations'].append(annotation)
            id += 1
        img_name_to_id[gt_img_path] = img_id
        img = cv2.imread(gt_img_path)
        coco_gt['images'].append(
            {'file_name': gt_img_path, 'id': img_id, 'height': img.shape[0], 'width': img.shape[1]})
        img_id += 1

    for pred_line in pred_lines:
        pred_line = pred_line.rstrip()
        pred_img_name = pred_line.split(' ')[0]
        pred_data = pred_line.split(' ')
        pred_vectors = pred_data[1:]

        for pred_vector in pred_vectors:
            pred_vector = str_list_to_float_list(pred_vector)
            annotation = {'image_id': img_name_to_id[pred_img_name],
                 'category_id': int(pred_vector[5]),
                 'bbox': [pred_vector[0], pred_vector[1], pred_vector[2] - pred_vector[0],
                          pred_vector[3] - pred_vector[1]],
                 'segmentation': [pred_vector[6:]],
                 'score': pred_vector[4]
                 }
            if len(pred_vector) > 6:
                annotation['segmentation'] = [pred_vector[6:]]
            coco_pred.append(annotation)
    with open('tmp_coco_pred.json', 'w') as ccp:
        json.dump(coco_pred, ccp)
    with open('tmp_coco_gt.json', 'w') as ccg:
        json.dump(coco_gt, ccg)
    print('conversion done successfully!')


def coco_eval(type):
    cocoGt = COCO('tmp_coco_gt.json')
    cocoDt = cocoGt.loadRes('tmp_coco_pred.json')
    cocoEval = COCOeval(cocoGt, cocoDt, type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def str_list_to_float_list(string):
    return list(map(float, string.split(',')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate mean average precision between predictions and ground truth.')
    parser.add_argument('pred', type=str, help='path to prediction text file')
    parser.add_argument('gt', type=str, help='path to ground truth text file')
    parser.add_argument('cls', type=str, help='path to classes text file, one class per row')
    parser.add_argument('type', type=str, choices=['bbox', 'segm'], help='whenever bounding boxes or polygons (segm) are evaluated')
    args = parser.parse_args()
    pred = args.pred
    gt = args.gt
    clss = args.cls
    type = args.type
    yolo_to_coco(pred, gt, clss)
    coco_eval(type)