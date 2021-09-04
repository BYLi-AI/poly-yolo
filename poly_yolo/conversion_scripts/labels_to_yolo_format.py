import argparse
import json
import os
import sys
import xml.etree.ElementTree as et

import numpy as np


def idd_class_to_number(cls):
    class_mapping = {'motorcycle': 0,
                     'autorickshaw': 1,
                     'car': 2,
                     'truck': 3,
                     'person': 4,
                     'animal': 5,
                     'rider': 6,
                     'vehicle fallback': 7,
                     'traffic sign': 8,
                     'bus': 9,
                     'bicycle': 10,
                     'caravan': 11,
                     'traffic light': 12,
                     'trailer': 13}
    if cls in class_mapping.keys():
        return class_mapping[cls]
    else:
        return None


def cityscape_class_to_number(cls):
    class_mapping = {'road': 0
                     }
    if cls in class_mapping.keys():
        return class_mapping[cls]
    else:
        return None


def convert_json(annotation_folder, image_folder, dataset):
    if dataset == 'idd':
        cls_chck = idd_class_to_number
    elif dataset == 'cityscapes':
        cls_chck = cityscape_class_to_number
    else:
        print('unsupported dataset')
    print('converting', dataset)
    print(image_folder)
    out = open('annotation.txt', 'w')
    skipped = 0
    img_type = None
    for root, subdirs, files in os.walk(annotation_folder):
        for file in files:
            if ".json" not in file:
                continue
            else:
                if os.path.exists(os.path.join(image_folder, root[len(annotation_folder) + 1:], file[:-21] + '_leftImg8bit.png')):
                    img_type = '.png'
                    break
                elif os.path.exists(os.path.join(image_folder, root[len(annotation_folder) + 1:], file[:-21] + '_leftImg8bit.jpg')):
                    img_type = '.jpg'
                    break
                elif os.path.exists(os.path.join(image_folder, root[len(annotation_folder) + 1:], file[:-21] + '_leftImg8bit.JPG')):
                    img_type = '.JPG'
                    break
                else:
                    print('image', file[:-21] + '_leftImg8bit.* in supported format does not exists')
    if img_type is None:
        print('Could not find any image in supported format')
        exit(0)

    for root, subdirs, files in os.walk(annotation_folder):
        print(root)
        for file in files:
            if ".json" not in file:
                continue
            with open(os.path.join(root, file)) as f:
                jf = json.load(f)
            objects = jf['objects']
            polygons_line = ''
            for object in objects:
                c = cls_chck(object['label'])
                # will fire only if label is not valid one
                if c is None:
                    continue
                polygon = object['polygon']
                min_x = sys.maxsize
                max_x = 0
                min_y = sys.maxsize
                max_y = 0
                polygon_line = ''
                for x, y in polygon:
                    if x > max_x: max_x = x
                    if y > max_y: max_y = y
                    if x < min_x: min_x = x
                    if y < min_y: min_y = y
                    polygon_line += ',{},{}'.format(x, y)
                if max_x - min_x <= 1.0 or max_y - min_y <= 1.0:
                    skipped += 1
                    continue
                polygons_line += ' {},{},{},{},{}'.format(min_x, min_y, max_x, max_y, c) + polygon_line
            if polygons_line == '': continue
            file = file[:-21] + '_leftImg8bit' + img_type
            annotation_line = os.path.join(image_folder, root[len(annotation_folder) + 1:], file) + polygons_line
            print(annotation_line, file=out)
    print('I have skipped total number of {} boxes due its width or height being <=1.0'.format(skipped))
    out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Labels in json format (cityscapes, IDD, etc.) to yolo labels file.'
                    'For the list of classes, open script and search for class_to_number method.')
    parser.add_argument('--labels_root', type=str,
                        help='path to dataset json annotation root folder, i.e. ...\\gtFine\\train')
    parser.add_argument('--images_root', type=str,
                        help='path to dataset images root folder, i.e. ...\\leftImg8bit\\train')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name (cityspaces or idd)')
    args = parser.parse_args()
    dataset_labels = args.labels_root
    dataset_images = args.images_root
    dataset_name = args.dataset

    if dataset_labels is not None and dataset_images is not None and dataset_name is not None:
        convert_json(dataset_labels, dataset_images, dataset_name)
    else:
        print('Please provide dataset_labels, dataset_images and dataset_name')
