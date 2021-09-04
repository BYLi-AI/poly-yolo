**Scripts for converting datasets to YOLO format**

Script [labels_to_yolo_format.py](labels_to_yolo_format.py) converts Cityscapes and IDD datasets annotations to YOLO annotation format. The script needs the following parameters to be set:

* '--images_root' - path to the folder with images
* '--labels_root' - path to the folder where are .json files
* '-d' - dataset name, either cityscapes or idd

Converted annotation is placed into --images_root folder. The script also has '--help' option.
