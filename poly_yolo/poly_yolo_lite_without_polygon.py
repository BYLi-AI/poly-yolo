#that is fork from https://github.com/qqwweee/keras-yolo3
#the modifications are as follows:
# - backbone uses squeeze-and-excitation blocks and has fewer parameters
# - the neck and head is new, it has single output scale
# - the instance segmentation is available in poly_yolo.py
#it accepts the same input data as poly_yolo

from datetime import datetime
import colorsys
import os
import sys
from functools import reduce
from functools import wraps

import math
import random as rd
import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers import Input, GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, Activation, add, Lambda, concatenate, MaxPooling2D
from keras.layers import Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adadelta, Adagrad
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

np.set_printoptions(precision=3, suppress=True)
MAX_VERTICES = 1000 #that allows the labels to have 1000 vertices per polygon at max. They are reduced for training
ANGLE_STEP  = 15 #that means Poly-YOLO will detect 360/15=24 vertices per polygon at max
NUM_ANGLES3  = 0
NUM_ANGLES  = 0

grid_size_multiplier = 8 #that is resolution of the output scale compared with input. So it is 1/8
anchor_mask = [[0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8]] #that should be optimized
anchors_per_level = 9 #single scale and nine anchors



def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw = image.shape[1]
    ih = image.shape[0]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    cvi = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cvi = cv.resize(cvi, (nw, nh), interpolation=cv.INTER_CUBIC)
    dx = int((w - nw) // 2)
    dy = int((h - nh) // 2)
    new_image = np.zeros((h, w, 3), dtype='uint8')
    new_image[...] = 128
    if nw <= w and nh <= h:
        new_image[dy:dy + nh, dx:dx + nw, :] = cvi
    else:
        new_image = cvi[-dy:-dy + h, -dx:-dx + w, :]

    return new_image.astype('float32') / 255.0


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(line, input_shape, random=True, max_boxes=80, hue_alter=20, sat_alter=30, val_alter=30, proc_img=True):
    # load data
    # the color conversion is later. it is not necessary to realize bgr->rgb->hsv->rgb
    image = cv.imread(line[0])
    iw = image.shape[1]
    ih = image.shape[0]
    h, w = input_shape
    box = np.array([np.array(list(map(float, box.split(',')[:5]))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            # image = image.resize((nw, nh), Image.BICUBIC)
            image = cv.cvtColor(
                cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC), cv.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.
        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box), 0:5] = box[:, 0:5]
        return image_data, box_data

    # resize image
    random_scale = rd.uniform(.6, 1.4)
    scale = min(w / iw, h / ih)
    nw = int(iw * scale * random_scale)
    nh = int(ih * scale * random_scale)

    # force nw a nh to be an even
    if (nw % 2) == 1:
        nw = nw + 1
    if (nh % 2) == 1:
        nh = nh + 1

    # jitter for slight distort of aspect ratio
    if np.random.rand() < 0.3:
        if np.random.rand() < 0.5:
            nw = int(nw*rd.uniform(.8, 1.0))
        else:
            nh = int(nh*rd.uniform(.8, 1.0))

    image = cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC)
    nwiw = nw/iw
    nhih = nh/ih

    # clahe. applied on resized image to save time. but before placing to avoid
    # the influence of homogenous background
    if np.random.rand() < 0.05:
        clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        cl = clahe.apply(l)
        limg = cv.merge((cl, a, b))
        image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    # place image
    dx = rd.randint(0, max(w - nw, 0))
    dy = rd.randint(0, max(h - nh, 0))

    new_image = np.full((h, w, 3), 128, dtype='uint8')
    new_image, crop_coords, new_img_coords = random_crop(
        image, new_image)

    # flip image or not
    flip = rd.random() < .5
    if flip:
        new_image = cv.flip(new_image, 1)

    # distort image
    hsv = np.int32(cv.cvtColor(new_image, cv.COLOR_BGR2HSV))

    # linear hsv distortion
    hsv[..., 0] += rd.randint(-hue_alter, hue_alter)
    hsv[..., 1] += rd.randint(-sat_alter, sat_alter)
    hsv[..., 2] += rd.randint(-val_alter, val_alter)

    # additional non-linear distortion of saturation and value
    if np.random.rand() < 0.5:
        hsv[..., 1] = hsv[..., 1]*rd.uniform(.7, 1.3)
        hsv[..., 2] = hsv[..., 2]*rd.uniform(.7, 1.3)

    hsv[..., 0][hsv[..., 0] > 179] = 179
    hsv[..., 0][hsv[..., 0] < 0] = 0
    hsv[..., 1][hsv[..., 1] > 255] = 255
    hsv[..., 1][hsv[..., 1] < 0] = 0
    hsv[..., 2][hsv[..., 2] > 255] = 255
    hsv[..., 2][hsv[..., 2] < 0] = 0

    image_data = cv.cvtColor(
        np.uint8(hsv), cv.COLOR_HSV2RGB).astype('float32') / 255.0

    # add noise
    if np.random.rand() < 0.15:
        image_data = np.clip(image_data + np.random.rand() *
                             image_data.std() * np.random.random(image_data.shape), 0, 1)

    # correct boxes
    box_data = np.zeros((max_boxes, 5))

    if len(box) > 0:
        np.random.shuffle(box)
        # rescaling separately because 5-th element is class
        box[:, [0, 2]] = box[:, [0, 2]] * nwiw
        box[:, [1, 3]] = box[:, [1, 3]] * nhih

        # transform boxes to new coordinate system w.r.t new_image
        box[:, :2] = box[:, :2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
        box[:, 2:4] = box[:, 2:4] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
        if flip:
            box[:, [0, 2]] = (w-1) - box[:, [2, 0]]

        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] >= w] = w-1
        box[:, 3][box[:, 3] >= h] = h-1
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        if len(box) > max_boxes:
            box = box[:max_boxes]

        box_data[:len(box), 0:5] = box[:, 0:5]
    return image_data, box_data


def random_crop(img, new_img):
    """Creates random crop from img and insert it into new_img

    Args:
        img (numpy array): Image to be cropped
        new_img (numpy array): Image to which the crop will be inserted into.

    Returns:
        tuple: Tuple of image containing the crop, list of coordinates used to crop img and list of coordinates where the crop
        has been inserted into in new_img
    """
    h, w = img.shape[:2]
    crop_shape = new_img.shape[:2]
    crop_coords = [0, 0, 0, 0]
    new_pos = [0, 0, 0, 0]
    # if image height is smaller than cropping window
    if h < crop_shape[0]:
        # cropping whole image [0,h]
        crop_coords[1] = h
        # randomly position whole img along height dimension
        val = rd.randint(0, crop_shape[0]-h)
        new_pos[0:2] = [val, val + h]
    else:
        # if image height is bigger than cropping window
        # randomly position cropping window on image
        crop_h_shift = rd.randint(crop_shape[0], h)
        crop_coords[0:2] = [crop_h_shift - crop_shape[0], crop_h_shift]
        new_pos[0:2] = [0, crop_shape[0]]

    # same as above for image width
    if w < crop_shape[1]:
        crop_coords[3] = w
        val = rd.randint(0, crop_shape[1] - w)
        new_pos[2:4] = [val, val + w]
    else:
        crop_w_shift = rd.randint(crop_shape[1], w)
        crop_coords[2:4] = [crop_w_shift - crop_shape[1], crop_w_shift]
        new_pos[2:4] = [0, crop_shape[1]]

    # slice, insert and return image including crop and coordinates used for cropping and inserting
    # coordinates are later used for boxes adjustments.
    new_img[new_pos[0]:new_pos[1], new_pos[2]:new_pos[3],
            :] = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], :]
    return new_img, crop_coords, new_pos


"""Poly-YOLO Model Defined in Keras."""
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        y = squeeze_excite_block(y)
        x = Add()([x, y])
    return x


# https://github.com/titu1994/keras-squeeze-excite-network/blob/master/keras_squeeze_excite_network/se_resnet.py
def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = LeakyReLU(alpha=0.1)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def _tensor_shape(tensor):
    return getattr(tensor, '_keras_shape')



def darknet_body(x):
    base = 4  # orig base = 8
    x = DarknetConv2D_BN_Leaky(base * 4, (3, 3))(x)
    x = resblock_body(x, base * 8, 1)
    x = resblock_body(x, base * 16, 2)
    x = resblock_body(x, base * 32, 8)
    small = x
    x = resblock_body(x, base * 64, 8)
    medium = x
    x = resblock_body(x, base * 128, 8)
    big = x
    return small, medium, big



def make_last_layers(x, num_filters, out_filters):
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create Poly-YOLO model CNN body in Keras."""
    small, medium, big = darknet_body(inputs)

    base = 4
    small  = DarknetConv2D_BN_Leaky(base*32, (1, 1))(small)
    medium = DarknetConv2D_BN_Leaky(base*32, (1, 1))(medium)
    big    = DarknetConv2D_BN_Leaky(base*32, (1, 1))(big)

    all = Add()([medium, UpSampling2D(2,interpolation='bilinear')(big)])
    all = Add()([small, UpSampling2D(2,interpolation='bilinear')(all)])



    num_filters = base*32

    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(all)

    all = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5 + NUM_ANGLES3), (1, 1)))(x)

    return Model(inputs, all)



def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = anchors_per_level
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(tf.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1], name='yolo_head/tile/reshape/grid_y'),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(tf.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1], name='yolo_head/tile/reshape/grid_x'),
                    [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1, name='yolo_head/concatenate/grid')
    grid = K.cast(grid, K.dtype(feats))
    feats = tf.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5 + NUM_ANGLES3], name='yolo_head/reshape/feats')

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))

    box_confidence      = K.sigmoid(feats[..., 4:5])
    box_class_probs     = K.sigmoid(feats[..., 5:5 + num_classes])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes




def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=80,
              score_threshold=.5,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    input_shape = K.shape(yolo_outputs)[1:3] * grid_size_multiplier
    boxes = []
    box_scores = []

    for l in range(1):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs,
                                                               anchors[anchor_mask[l]], num_classes, input_shape,
                                                               image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5+69)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape
        vstup je to nase kratke
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # tady jsou uz stredy
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]


    true_boxes[:,:, 5:NUM_ANGLES3 + 5:3] /= np.clip(np.expand_dims(np.sqrt(np.power(boxes_wh[:, :, 0], 2) + np.power(boxes_wh[:, :, 1], 2)), -1), 0.0001, 9999999)

    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: grid_size_multiplier}[l] for l in range(1)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes + NUM_ANGLES3),
                       dtype='float32') for l in range(1)]


    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0


    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            l = 0
            if n in anchor_mask[l]:
                i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                k = anchor_mask[l].index(n)
                c = true_boxes[b, t, 4].astype('int32')

                y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                y_true[l][b, j, i, k, 4] = 1
                y_true[l][b, j, i, k, 5 + c] = 1
    return y_true


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    """Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = 1
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    g_y_true = y_true
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * grid_size_multiplier, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0

    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    for layer in range(num_layers):
        object_mask = y_true[layer][..., 4:5]
        true_class_probs = y_true[layer][..., 5:5 + num_classes]

        grid, raw_pred, pred_xy, pred_wh= yolo_head(yolo_outputs[layer], anchors[anchor_mask[layer]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        raw_true_xy = y_true[layer][..., :2] * grid_shapes[layer][::-1] - grid

        raw_true_wh = K.log(y_true[layer][..., 2:4] / anchors[anchor_mask[layer]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf

        box_loss_scale = 2 - y_true[layer][..., 2:3] * y_true[layer][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[layer][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                                                                                                             from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:5 + num_classes], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        class_loss = K.sum(class_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf


        loss += (xy_loss + wh_loss + confidence_loss + class_loss)/ (K.sum(object_mask) + 1)*mf
    return loss


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'yolo_anchors.txt',
        "classes_path": 'yolo_classes.txt',
        "score": 0.2,
        "iou": 0.5,
        "model_image_size": (352, 608),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes= self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), anchors_per_level, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            # novy output
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5 + NUM_ANGLES3), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                                     len(self.class_names), self.input_image_shape,
                                                     score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        # start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            print('THE functionality is not implemented!')

        image_data = np.expand_dims(boxed_image, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })

        return out_boxes, out_scores, out_classes

    def close_session(self):
        self.sess.close()


if __name__ == "__main__":

    """
    Retrain the YOLO model for your own dataset.
    """


    def _main():
        phase = 1
        annotation_path = 'train.txt'
        validation_path = 'val.txt'
        log_dir = 'models/'
        classes_path = 'classes.txt'
        anchors_path = 'yolo_anchors.txt'
        class_names = get_classes(classes_path)
        num_classes = len(class_names)
        anchors = get_anchors(anchors_path)

        input_shape = (320,608) # multiple of 32, hw

        if phase == 1:
            model = create_model(input_shape, anchors, num_classes, load_pretrained=False)
        else:
            model = create_model(input_shape, anchors, num_classes, load_pretrained=True, weights_path=log_dir+'pretrained_model.h5')
        print(model.summary())

        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True, period=1, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, delta=0.03)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        with open(annotation_path) as f:
            lines = f.readlines()

        with open(validation_path) as f:
            lines_val = f.readlines()

        for i in range (0, len(lines)):
            lines[i] = lines[i].split()

        for i in range(0, len(lines_val)):
            lines_val[i] = lines_val[i].split()


        num_val = int(len(lines_val))
        num_train = len(lines)


        batch_size = 8  # decrease/increase batch size according to your memory of your GPU
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        from numpy.random import seed
        from tensorflow import set_random_seed
        np.random.seed(1)
        seed(1)
        set_random_seed(1)


        model.compile(optimizer=Adadelta(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        epochs = 100
        history = model.fit_generator(data_generator_wrapper(lines, batch_size, input_shape, anchors, num_classes, True),
                                      steps_per_epoch=max(1, num_train // batch_size),
                                      validation_data=data_generator_wrapper(lines_val, batch_size, input_shape, anchors, num_classes, False),
                                      validation_steps=max(1, num_val // batch_size),
                                      epochs=epochs,
                                      initial_epoch=0,
                                      callbacks=[reduce_lr, early_stopping, checkpoint])

    def get_classes(classes_path):
        """loads the classes"""
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def get_anchors(anchors_path):
        """loads the anchors from a file"""
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                     weights_path='model_data/yolo_weights.h5'):
        """create the training model"""
        K.clear_session()  # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = input_shape
        num_anchors = len(anchors)
        y_true = Input(shape=(h // grid_size_multiplier, w // grid_size_multiplier, anchors_per_level, num_classes + 5 + NUM_ANGLES3))

        model_body = yolo_body(image_input, anchors_per_level, num_classes)
        print('Create Poly-YOLO model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [model_body.output, y_true])
        model = Model([model_body.input, y_true], model_loss)

        # print(model.summary())
        return model


    def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, is_random):
        """data generator for fit_generator"""
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i == 0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=is_random)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)


    def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0: return None
        return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random)


    if __name__ == '__main__':
        _main()
