import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['CMU']
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy.matlib as mx
import matplotlib.colors as mcolors

num_precision = tf.keras.backend.floatx()

colorlist = list({**mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS}.keys())[1:]


def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)

def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )

def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )

def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

# def visualize_detections(
#     image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
# ):
#     """Visualize Detections"""
#     image = np.array(image, dtype=np.uint8)
#     plt.figure(figsize=figsize)
#     plt.axis("off")
#     plt.imshow(image)
#     ax = plt.gca()
#     for box, _cls, score in zip(boxes, classes, scores):
#         text = "{}: {:.2f}".format(_cls, score)
#         x1, y1, x2, y2 = box
#         w, h = x2 - x1, y2 - y1
#         patch = plt.Rectangle(
#             [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
#         )
#         ax.add_patch(patch)
#         ax.text(
#             x1,
#             y1,
#             text,
#             bbox={"facecolor": color, "alpha": 0.4},
#             clip_box=ax.clipbox,
#             clip_on=True,
#         )
#     plt.show()
#     return ax

# def visualize_datasample(
#     image, boxes, classes, figsize=(7, 7), linewidth=1, color=[0, 0, 1], showlabels=True
# ):
#     """Visualize Data"""
#     image = np.array(image, dtype=np.uint8)
#     plt.figure(figsize=figsize)
#     plt.axis("off")
#     plt.imshow(image)
#     ax = plt.gca()
#     for box, _cls in zip(boxes, classes):
#         text = "{}".format(_cls)
#         x1, y1, x2, y2 = box
#         w, h = x2 - x1, y2 - y1
#         patch = plt.Rectangle(
#             [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
#         )
#         ax.add_patch(patch)
#         if showlabels:
#             ax.text(
#                 x1,
#                 y1,
#                 text,
#                 bbox={"facecolor": color, "alpha": 0.4},
#                 clip_box=ax.clipbox,
#                 clip_on=True,
#             )
#     plt.show()
#     return ax

def visualize_datasample(
    image, boxes, classes, colors=None, figsize=(9, 9), linewidth=1.5, showlabels=False
):
    """Visualize Data"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    if not colors: colors=colorlist

    for box, _cls in zip(boxes, classes):
        text = "{}".format(_cls)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=colors[int(_cls)], linewidth=linewidth
        )
        ax.add_patch(patch)
        if showlabels:
            ax.text(
                x1,
                y1,
                text,
                bbox={"facecolor": colors[int(_cls)], "alpha": 0.4},
                clip_box=ax.clipbox,
                clip_on=True,
            )
    plt.show()
    return ax

def visualize_detections(
    image, boxes, class_ids, scores, class_labels=None, colors=None, figsize=(9, 9), linewidth=1.5, showscores=True
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    if not colors: colors=colorlist
    i=0
    for box, _cls, score in zip(boxes, class_ids, scores):
        if class_labels and showscores: text = "{}: {:.2f}".format(class_labels[i], score)
        elif class_labels: text = "{}".format(class_labels[i])
        elif showscores: text = "{:.2f}".format(score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=colors[int(_cls)], linewidth=linewidth
        )
        ax.add_patch(patch)
        if class_labels or showscores:
            ax.text(
                x1,
                y1,
                text,
                fontdict={"Fontsize": 10},
                bbox={"facecolor": colors[int(_cls)], "alpha": 0.4},
                clip_box=ax.clipbox,
                clip_on=True,
            )
        i+=1
    plt.show()
    return ax

def compare_detections(
    image, boxes, classes, figsize=(7, 7), linewidth=1, color=[[0,0,1],[1,0,0]]
):
    """Visualize Data"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for i in range(2):
        for box, _cls in zip(boxes[i], classes[i]):
            text = "{}".format(_cls)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            patch = plt.Rectangle(
                [x1, y1], w, h, fill=False, edgecolor=color[i], linewidth=linewidth
            )
            ax.add_patch(patch)
            ax.text(
                x1,
                y1,
                text,
                bbox={"facecolor": color[i], "alpha": 0.4},
                clip_box=ax.clipbox,
                clip_on=True,
            )
    plt.show()
    return ax



class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self, ShippingLab=False):
        if ShippingLab:
            self.aspect_ratios = [0.8, 1.4, 2.6]
            self.scales = [2 ** 0, 2 ** (2.0 / 3.0), 2 ** (-2.0 / 3.0)]
            #self.aspect_ratios = [0.5, 1.0, 2.0]
            #self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
            self._num_anchors = len(self.aspect_ratios) * len(self.scales)
            self._strides = [8, 16, 32, 64, 128]
            #self._areas = [x ** 2 for x in [11.0, 21.0, 40.0, 77.0, 147.0, 283.0, 542.0]]
            self._areas = [x ** 2 for x in  [13.83181749,  21.00875367,  34.26314969,  59.80681257, 131.65011644, 320.0]]
        else:
            self.aspect_ratios = [0.5, 1.0, 2.0]
            self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
            self._num_anchors = len(self.aspect_ratios) * len(self.scales)
            self._strides = [2 ** i for i in range(3, 8)]
            self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()


    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=num_precision) + 0.5
        ry = tf.range(feature_height, dtype=num_precision) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)

def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes

def resize_and_pad_image(image, tar_shape = [896,1152], jitter=None):
    
    image_shape = tf.cast(tf.shape(image)[:2], dtype=num_precision)
    ratio = tar_shape[0] / image_shape[0]
    if ratio*image_shape[1] > tar_shape[1]:
        ratio = tar_shape[1] / image_shape[1]
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, tar_shape[0], tar_shape[1]
    )
    return image, image_shape, ratio

def resize_and_crop_image(image, tar_shape = [896,1280],  #[1024,1536], #
    jitter=None):
    
    image_shape = tf.cast(tf.shape(image)[-3:-1], dtype=num_precision)
    ratio = tar_shape[1] / image_shape[1]
    if ratio*image_shape[0] < tar_shape[0]:
        ratio = tar_shape[0] / image_shape[0]
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    image = tf.image.crop_to_bounding_box(
        image, 0, 0, tar_shape[0], tar_shape[1]
    )
    return image, image_shape, ratio

# def resize_and_pad_image(image, tar_shape = [896,1152], jitter=None, pad=128*2):
    
#     tar_shape = np.array(tar_shape)-pad
#     image_shape = tf.cast(tf.shape(image)[:2], dtype=num_precision)
#     ratio = tar_shape[0] / image_shape[0]
#     if ratio*image_shape[1] > tar_shape[1]:
#       ratio = tar_shape[1] / image_shape[1]
#     image_shape = ratio * image_shape
#     image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
#     image = tf.image.pad_to_bounding_box(
#         image, pad, pad, tar_shape[0]+2*pad, tar_shape[1]+2*pad
#     )
#     return image, image_shape, ratio

# def resize_and_pad_image(
#     image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
# ):
#     """Resizes and pads image while preserving aspect ratio.

#     1. Resizes images so that the shorter side is equal to `min_side`
#     2. If the longer side is greater than `max_side`, then resize the image
#       with longer side equal to `max_side`
#     3. Pad with zeros on right and bottom to make the image shape divisible by
#     `stride`

#     Arguments:
#       image: A 3-D tensor of shape `(height, width, channels)` representing an
#         image.
#       min_side: The shorter side of the image is resized to this value, if
#         `jitter` is set to None.
#       max_side: If the longer side of the image exceeds this value after
#         resizing, the image is resized such that the longer side now equals to
#         this value.
#       jitter: A list of floats containing minimum and maximum size for scale
#         jittering. If available, the shorter side of the image will be
#         resized to a random value in this range.
#       stride: The stride of the smallest feature map in the feature pyramid.
#         Can be calculated using `image_size / feature_map_size`.

#     Returns:
#       image: Resized and padded image.
#       image_shape: Shape of the image before padding.
#       ratio: The scaling factor used to resize the image
#     """
#     image_shape = tf.cast(tf.shape(image)[:2], dtype=num_precision)
#     if jitter is not None:
#         min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=num_precision)
#     ratio = min_side / tf.reduce_min(image_shape)
#     if ratio * tf.reduce_max(image_shape) > max_side:
#         ratio = max_side / tf.reduce_max(image_shape)
#     image_shape = ratio * image_shape
#     image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
#     padded_image_shape = tf.cast(
#         tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
#     )
#     image = tf.image.pad_to_bounding_box(
#         image, 0, 0, padded_image_shape[0], padded_image_shape[1]
#     )
#     return image, image_shape, ratio


def preprocess_data(sample, return_ratio=False, rand_flip=True):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    if rand_flip: image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, ratio = resize_and_pad_image(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    if return_ratio: return image, bbox, class_id, ratio
    return image, bbox, class_id

class preprocess_data_ShippingLab:
    def __init__(self, rand_flip=True, return_ratio=False, tar_shape = [896,1280]):
        self.rand_flip = rand_flip
        self.return_ratio = return_ratio
        self.tar_shape = tar_shape

    def call(self, sample, rand_flip=True, return_ratio=False):
        
        image = sample["image"]
        bbox = sample["objects"]["bbox"]
        class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

        if self.rand_flip or rand_flip: image, bbox = random_flip_horizontal(image, bbox)
        image, image_shape, ratio = resize_and_crop_image(image, tar_shape=self.tar_shape)

        bbox = tf.stack(
            [
                bbox[:, 0] * image_shape[1],
                bbox[:, 1] * image_shape[0],
                bbox[:, 2] * image_shape[1],
                bbox[:, 3] * image_shape[0],
            ],
            axis=-1,
        )
        bbox = convert_to_xywh(bbox)
        if self.return_ratio or return_ratio: return image, bbox, class_id, ratio
        return image, bbox, class_id

class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self, 
            ShippingLab=False,
            use_empty_samples=True, 
            resnet_preprocess=True,
            match_iou=0.5, 
            ignore_iou=0.4
        ):
        self._anchor_box = AnchorBox(ShippingLab=ShippingLab)
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=num_precision
        )
        self.match_iou = match_iou
        self.ignore_iou = ignore_iou
        self.use_empty_samples = use_empty_samples
        self.preprocess = resnet_preprocess

    def _match_anchor_boxes(self, anchor_boxes, gt_boxes):
        match_iou = self.match_iou
        ignore_iou = self.ignore_iou
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=num_precision),
            tf.cast(ignore_mask, dtype=num_precision),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        if self.use_empty_samples and tf.reduce_max(cls_ids)==0: 
            gt_boxes = tf.convert_to_tensor([[0.0, 0.0, 1.0, 1.0]], dtype=num_precision)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=num_precision)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=num_precision, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        if self.preprocess: batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()

class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        batch_size = 1,
        ShippingLab=False,
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox(ShippingLab=ShippingLab)
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=num_precision
        )
        self.batch_size = batch_size

    def adapt(self, a):
        return tf.reshape(a,[self.batch_size,-1,self.num_classes+4])

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        predictions = self.adapt(predictions)
        image_shape = tf.cast(tf.shape(images), dtype=num_precision)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])

        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], predictions[:, :, :4])

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0, batch_size=1):
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes
        self.batch_size = batch_size

    def adapt(self, a):
        return tf.reshape(a,[self.batch_size,-1,self._num_classes+4])

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=num_precision)
        box_labels = y_true[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=num_precision,
        )
        y_pred = self.adapt(y_pred)
        box_predictions = y_pred[:, :, :4]
        cls_predictions = y_pred[:, :, 4:]

        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=num_precision)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=num_precision)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss


class RetinaNetLoss_Norm(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, norm, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0, batch_size=1):
        super(RetinaNetLoss_Norm, self).__init__(reduction="auto", name="RetinaNetLoss_Norm")
        self.scalefactors = norm
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes
        self.batch_size = batch_size
        #norm = tf.convert_to_tensor(norm)

    def adapt(self, a):
        return tf.reshape(a,[self.batch_size,-1,self._num_classes+4])

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=num_precision)
        box_labels = y_true[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=num_precision,
        )
        # Undo normalization
        y_pred = y_pred*(self.scalefactors[0]-self.scalefactors[1]) + self.scalefactors[1]
        y_pred = self.adapt(y_pred)
        box_predictions = y_pred[:, :, :4]
        cls_predictions = y_pred[:, :, 4:]

        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=num_precision)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=num_precision)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss

    def set_norm(self, norm):
        self.scalefactors = norm


class DecodeNormalized(tf.keras.layers.Layer):
    def __init__(self, lmbda_shift=None, numerical_precision = tf.float32, norm=None, **kwargs):
        super(DecodeNormalized, self).__init__(**kwargs) 
        self.norm_init = lmbda_shift
        self.norm_adapted = False
        self.prec = numerical_precision
        self.norm = norm

    def build(self, input_shape): 
        if self.norm is None:
            self.norm = [self.add_weight(
                            name="lambda",
                            shape = input_shape,
                            initializer = "ones", trainable = False
                            ),
                        self.add_weight(
                            name="shift",
                            shape = input_shape,
                            initializer = "zeros", trainable = False
                            )
            ]
        else: 
            self.norm_adapted = True
        
        if self.norm_init is not None and not self.norm_adapted:
            self.adapt_norm(input_shape)
        super(DecodeNormalized, self).build(input_shape)

    def call(self, input_data):
        out = tf.convert_to_tensor(input_data)
        out = tf.cast(out, dtype=self.prec)
        if not self.norm_adapted:
            self.adapt_norm(tf.shape(out))
        out = out*(self.norm[0]-self.norm[1]) + self.norm[1]
        return tf.cast(out, dtype=num_precision)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'norm': self.norm,
            'lmbda_shift': self.norm_init,
            'numerical_precision': self.prec
        })
        return config

    def adapt_norm(self, shape):
        n = [0, 4, shape[-1]-4]
        new_norm = [[tf.convert_to_tensor(lbdashft, dtype=self.prec) for lbdashft in boxcls] \
            for boxcls in self.norm_init]
        aux = [[] for boxcls in self.norm_init]
        

        for i,item in enumerate(new_norm):
            for j in range(2):
                w = item[j]
                w = tf.reshape(w,[shape[0],1,-1,n[i+1]])
                r = int(shape[-2]/tf.shape(w)[-2])
                w = tf.tile(w, [1,1,r,1])
                aux[j].append(w)
        
        self.norm = [tf.concat(item, axis=-1) for item in aux]
        self.norm_adapted = True

def adapt_norm_to_output(norm_init, shape):
    n = [0, 4, shape[-1]-4]
    new_norm = [[tf.convert_to_tensor(lbdashft, dtype=num_precision) for lbdashft in boxcls] \
        for boxcls in norm_init]
    aux = [[] for boxcls in norm_init]
    

    for i,item in enumerate(new_norm):
        for j in range(2):
            w = item[j]
            w = tf.reshape(w,[shape[0],1,-1,n[i+1]])
            r = int(shape[-2]/tf.shape(w)[-2])
            w = tf.tile(w, [1,1,r,1])
            aux[j].append(w)
    
    norm = tuple([tf.concat(item, axis=-1) for item in aux])
    return norm

# class MeanAveragePrecision(tf.keras.metrics.Metric):
#     def __init__(self, name='mAP', **kwargs):
#         super(MeanAveragePrecision, self).__init__(name=name, **kwargs)
#         self.accum_AP = self.add_weight(name='mAP', initializer='zeros')
#         self.num_samples = 0

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         self.num_samples+=1
#         # k = y_true.get_shape()[-2]
#         # if k is None: k=1
#         k=3
#         AP = tf.compat.v1.metrics.average_precision_at_k(y_true, y_pred, k)
#         self.accum_AP.assign_add(AP)

#     def result(self):
#         return self.accum_AP/self.num_samples

#     def reset_states(self):
#         self.accum_AP.assign(0)
#         self.num_samples = 0

class MeanAveragePrecision():
    def __init__(self, num_classes=80, iou_thresh=0.5, init_img_idx=None):
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.epsilon = 1e-6

        # States
        self.det_table = [tf.convert_to_tensor([]) for _ in range(self.num_classes)] # [[img_idx, gt_idx, score, iou]...]
        self.img_idx = 0
        self.img_gts_checkbox = []
        self.gts_per_class = [0 for _ in range(self.num_classes)]
        if init_img_idx: self.img_idx=init_img_idx

    def set_states(self, det_table, img_gts_checkbox, gts_per_class):
        self.det_table = det_table
        self.img_gts_checkbox = img_gts_checkbox
        self.gts_per_class = gts_per_class


    def update_metrics(self, y_ref, y_det, save_det_path=None): # runs for every image in the dataset
        """     
        y_det = [classes, scores, x0, y0, x1, y1]
        y_ref  = [classes, x0, y0, x1, y1]

        """
        self.img_gts_checkbox.append([False]*len(y_ref))
        for c in y_ref[:,0]:
            self.gts_per_class[c.numpy().astype('int')] += 1 

        if len(y_det) and len(y_ref):
            iou_matrix = compute_iou(y_det[:,-4:], y_ref[:,-4:])

            det_iou = tf.reduce_max(iou_matrix, axis=1).numpy()
            det_match = tf.argmax(iou_matrix, axis=1).numpy()

            correct_class = [float(y_ref[int(ix),0]==pred_cls) for ix,pred_cls in zip(det_match,y_det[:,0])]

            det = tf.transpose(tf.convert_to_tensor([
                    [self.img_idx]*len(y_det), # img_idx
                    det_match,                 # gt_idx
                    y_det[:,1],                # score
                    det_iou,                   # iou
                    correct_class,             # correctness of the chosen class
                    y_det[:,0]                 # classes
                ]))
            # [img_idx, gt_idx, score, iou, classes]
            for c in range(self.num_classes):
                cls_det = tf.convert_to_tensor([row[:5] for row in det if row[5]==c])
                if tf.size(cls_det):
                    self.det_table[c] = tf.concat(
                            [
                                self.det_table[c], 
                                cls_det
                            ], axis=0
                        ) if tf.size(self.det_table[c]) else cls_det

        if save_det_path:
            import json
            states = {}
            states['det_table'] =  [det.numpy().tolist() for det in self.det_table]
            states['img_idx'] = self.img_idx
            states['img_gts_checkbox'] = self.img_gts_checkbox
            states['gts_per_class'] = self.gts_per_class
            #np.savez_compressed(os.path.join(save_det_path), self.det_table)
            with open(save_det_path, str('w')) as f:
                json.dump(states, f)

        self.img_idx+=1

        
        # display(iou)
        # display(det_iou)
        # display(det_match)
        # display(best_score)
        # display(best_match)
        #display(self.det_table)

    def result(self, iou_thresh=None, show_plots=False, save_path=None, possible_labels=None):
        if not iou_thresh: iou_thresh=self.iou_thresh
        self.reset_checkbox()

        self.Precision = np.zeros(self.num_classes)
        self.Recall = np.zeros(self.num_classes)
        self.AP = np.zeros(self.num_classes)

        if not possible_labels: possible_labels=range(self.num_classes)

        for c in possible_labels:
            detections = self.det_table[c]
            if not isinstance(detections, list):
                detections = detections.numpy().tolist() # Detections for that class
            detections.sort(key=lambda x: x[2], reverse=True) # Sort detections by score

            # Metrics
            TP = 0
            FP = 0
            precision = [0]*len(detections)
            recall = [0]*len(detections)

            for i,detection in enumerate(detections):
                # detection = [img_idx, gt_idx, score, iou]
                if detection[3]>iou_thresh and detection[-1] and not self.img_gts_checkbox[int(detection[0])][int(detection[1])]: # if iou over threshold, correct class and this gt has not been detected before
                    TP+=1
                    self.img_gts_checkbox[int(detection[0])][int(detection[1])]=True
                else:
                    FP+=1
                precision[i] = TP/(TP+FP)
                recall[i] = TP/self.gts_per_class[c] if self.gts_per_class[c] else 0
            if precision: 
                precision = [precision[0]]+precision
                recall = [0]+recall
            self.Precision[c] = precision[-1] if precision else None
            self.Recall[c] = recall[-1] if recall else None
            self.AP[c] = np.trapz(precision, recall)

            if show_plots or save_path:
                print(c)
                print('mAP = ',self.AP[c])
                fig, ax = plt.subplots(figsize=(4,4))
                ax.fill_between(recall, precision, color="skyblue", alpha=0.2)
                ax.plot(recall, precision, color="Slateblue", alpha=0.6)
                ax.set_ylabel('Precision')
                ax.set_xlabel('Recall')
                ax.set_ylim([0,1])
                ax.set_xlim([0,1])
                ax.set_title('Precision/Recall graph: #{}   AP = {:.2f}%'.format(c, self.AP[c]*100))
                ax.annotate(
                    'TPs={} | FPs={}\ngts={}'.format(TP,FP,self.gts_per_class[c]), xy=[1,1],  xycoords='data',
                    horizontalalignment='right', verticalalignment='top', 
                    color='black', fontsize=10, style='italic'
                    )
                if save_path:
                    plt.savefig(os.path.join(save_path,'PR{}.png'.format(c)), bbox_inches='tight')
                if show_plots: plt.show()
                else: plt.close(fig)
                print('\n')

        AP = [self.AP[label] for label in possible_labels]
        return np.sum(AP)/len(AP), AP

    def reset_checkbox(self):
        if self.img_gts_checkbox:
            self.img_gts_checkbox = [[False]*len(item) for item in self.img_gts_checkbox]


class DecodeTrainingData(tf.keras.layers.Layer):
   
    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        batch_size = 1,
        ShippingLab=False,
        **kwargs
    ):
        super(DecodeTrainingData, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox(ShippingLab=ShippingLab)
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=num_precision
        )
        self.batch_size = batch_size

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=num_precision)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        ps = predictions.shape
        cls_predictions = np.zeros([1,ps[1],self.num_classes], dtype='float32')
        o_idx = tf.squeeze(tf.where(tf.greater(predictions[0, :, 4], -1.0))).numpy()
        if not hasattr(o_idx, '__len__'): o_idx = [o_idx]
        for idx in o_idx:
            cls_predictions[0,idx,int(predictions[0,idx,4])] = 1.0
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], predictions[:, :, :4])

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )

def view_sample(sample, showlabels=True):
    x = tf.cast(sample["image"], dtype=tf.float32)
    image_shape = tf.cast(tf.shape(x)[-3:-1], dtype=tf.float32)
    #ybox = sample["objects"]["bbox"]
    #ybox = swap_xy(sample[1])
    ybox = convert_to_corners(sample[1])
    ycls = tf.cast(sample["objects"]["label"], dtype=tf.float32)
    ybox = tf.stack( # [classes, x0, y0, x1, y1]
        [
            ybox[..., 0] * image_shape[1],
            ybox[..., 1] * image_shape[0],
            ybox[..., 2] * image_shape[1],
            ybox[..., 3] * image_shape[0],
        ],
        axis=-1)

    visualize_datasample(x,ybox,ycls, showlabels=showlabels)

def view_sample2(sample, swapxy=False, showlabels=True):
    x = tf.cast(sample["image"], dtype=tf.float32)
    image_shape = tf.cast(tf.shape(x)[-3:-1], dtype=tf.float32)
    ybox = sample["objects"]["bbox"]
    if swapxy: ybox = swap_xy(ybox)
    #ybox = convert_to_corners(sample[1])
    ycls = tf.cast(sample["objects"]["label"], dtype=tf.float32)
    ybox = tf.stack( # [classes, x0, y0, x1, y1]
        [
            ybox[..., 0] * image_shape[1],
            ybox[..., 1] * image_shape[0],
            ybox[..., 2] * image_shape[1],
            ybox[..., 3] * image_shape[0],
        ],
        axis=-1)

    visualize_datasample(x,ybox,ycls, colors=colorlist, showlabels=showlabels)

