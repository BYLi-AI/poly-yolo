#original code @ https://github.com/qqwweee/keras-yolo3/blob/master/coco_annotation.py

import os
import json

def cls_chck(cat):
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat

id_img = {}
id_ann = {}
# without trailing dir separator!
img_folder = "coco/imgs/train2014"
output_file = "coco/yolo_labels.txt"
ann_file = "coco/instances_train2014.json"
f_out = open(output_file, 'w+')
with open(ann_file) as f_ann:
    jf = json.load(f_ann)
for img in jf['images']:
    id_img[img['id']] = img['file_name']
for ann in jf['annotations']:
    bbox = ann['bbox']
    # convert x,y,w,h to x1,y1,x2,y2
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    bbox = map(int, bbox)
    bbox = ','.join(map(str,bbox)).strip('[]')
    bbox = bbox.replace(' ', '')
    seg = ','.join(map(str, ann['segmentation'])).strip('[]')
    seg = seg.replace(' ', '')
    seg = seg.replace('[', '')
    seg = seg.replace(']', '')
    if ann['image_id'] in id_ann.keys():
        id_ann[ann['image_id']] += [bbox,  str(cls_chck(ann['category_id'])), seg]
    else:
        id_ann[ann['image_id']] = [bbox, str(cls_chck(ann['category_id'])), seg]
    
for img in id_img.keys():
    if img not in id_ann.keys():
        print('image ' + id_img[img] + ' does not have any annotation!')
        continue
    ann_line = ''
    ann_line += os.path.join(img_folder, id_img[img])
    for i in range(0, len(id_ann[img]), 3):
        ann_line += ' '+id_ann[img][i]+','+id_ann[img][i+1]+','+id_ann[img][i+2]
    print(ann_line, file=f_out)
f_out.close()
