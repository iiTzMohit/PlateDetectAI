import numpy as np
import cv2
import tensorflow as tf # 2.17.0
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Add, concatenate, Lambda
from tensorflow.keras.models import Model
import struct
import random

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, anchors,thresh_iou = 0.5, name='custom_loss',alpha = 0.75, gamma = 1.0, box_weight = 5.0):
        super(CustomLoss, self).__init__(name=name)
        self.anchors = anchors
        self.thresh_iou = thresh_iou
        self.gamma = gamma
        self.alpha = alpha
        self.box_weight = box_weight

    def yolo_head(self, feats, anchors):
        """Convert final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        grid_size = tf.shape(feats)[1]  # height, width
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=feats.dtype), [1, 1, 1, num_anchors, 2])

        # Reshape to (batch, height, width, num_anchors, box_params)
        feats = tf.reshape(feats, [-1, grid_size, grid_size, num_anchors, 5])

        # Grid
        grid_x = tf.range(grid_size, dtype=feats.dtype)
        grid_y = tf.range(grid_size, dtype=feats.dtype)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        grid = tf.stack([grid_x, grid_y], axis=-1)
        grid = tf.expand_dims(grid, axis=2)
        grid = tf.expand_dims(grid, axis=0)

        # Adjust predictions to each spatial grid point and anchor size
        box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_size, feats.dtype)
        box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor
        box_confidence = feats[..., 4:]

        return box_xy, box_wh, box_confidence


    def call(self, y_true, y_pred):
        # Split the predictions into individual components
        pred_box_xy, pred_box_wh, pred_conf = self.yolo_head(y_pred, self.anchors)

        true_box_xy = y_true[..., 0:2] # (x, y)
        true_box_wh = y_true[..., 2:4]  # (w, h)
        true_conf = y_true[..., 4:]  # objectness score

        # Compute IoU between predicted boxes and ground truth boxes
        pred_box_wh_half = pred_box_wh / 2.0
        pred_box_mins = pred_box_xy - pred_box_wh_half
        pred_box_maxes = pred_box_xy + pred_box_wh_half

        true_box_wh_half = true_box_wh / 2.0
        true_box_mins = true_box_xy - true_box_wh_half
        true_box_maxes = true_box_xy + true_box_wh_half

        intersect_mins = tf.maximum(pred_box_mins, true_box_mins)
        intersect_maxes = tf.minimum(pred_box_maxes, true_box_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

        union_area = pred_box_area + true_box_area - intersect_area
        iou_scores = intersect_area / tf.maximum(union_area, 1e-10)
        
        object_mask = tf.cast(true_conf > 0, tf.float32)
        no_object_mask = tf.reshape(tf.cast(iou_scores < self.thresh_iou, tf.float32), tf.shape(true_conf))

        # Calculate losses
        xy_loss = self.box_weight*tf.reduce_sum(tf.square(true_box_xy - pred_box_xy)*object_mask, axis=-1)
        wh_loss = self.box_weight*tf.reduce_sum(tf.square(tf.sqrt(true_box_wh) - tf.sqrt(pred_box_wh))*object_mask, axis=-1)
        conf_loss = tf.expand_dims(tf.keras.losses.binary_focal_crossentropy(true_conf, pred_conf, from_logits = True, alpha = self.alpha, gamma = self.gamma),-1)
        conf_loss_obj = object_mask*conf_loss
        conf_loss_noobj = (1-object_mask)*no_object_mask*conf_loss
        conf_loss_total = tf.reduce_sum(conf_loss_obj+conf_loss_noobj, axis = -1)
        # Sum all losses
        total_loss = tf.reduce_sum(xy_loss + wh_loss + conf_loss_total, axis=[1, 2, 3])

        # Compute the number of grid cells
        grid_size = tf.shape(y_true)[1:3]  # Grid size (height, width)
        num_grid_cells = tf.reduce_prod(grid_size) * tf.shape(y_true)[-2]  # Grid size * number of anchors

        # Normalize the total loss by the number of grid cells
        normalized_loss = tf.reduce_mean(total_loss) / tf.cast(num_grid_cells, tf.float32)  # Scalar

        return normalized_loss


def _conv_block(inp, convs, skip=True):
    x = inp
    count = 0

    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x
        count += 1

        if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
        x = Conv2D(conv['filter'],
                   conv['kernel'],
                   strides=conv['stride'],
                   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
                   name='conv_' + str(conv['layer_idx']),
                   use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
        if conv['leaky']: x = LeakyReLU(negative_slope=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    return Add()([skip_connection, x]) if skip else x


def make_yolov3_model():
    input_image = Input(shape=(None, None, 3), dtype=tf.float32)
    inputs = Lambda(lambda x: x / 255.0)(input_image)

    # Layer  0 => 4
    x = _conv_block(inputs, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])

    skip_36 = x

    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])

    skip_61 = x

    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])

    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)

    # Layer 80 => 82
    yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                              {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)

    # Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)

    # Layer 92 => 94
    yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                              {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)

    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    # Layer 99 => 106
    yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                               {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)

    model = Model(input_image, [yolo_82, yolo_94, yolo_106])
    return model


class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)

            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                print("loading weights of convolution #" + str(i))

                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('bnorm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta  = self.read_bytes(size) # bias
                    gamma = self.read_bytes(size) # scale
                    mean  = self.read_bytes(size) # mean
                    var   = self.read_bytes(size) # variance

                    weights = norm_layer.set_weights([gamma, beta, mean, var])

                if len(conv_layer.get_weights()) > 1:
                    bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print("no convolution #" + str(i))

    def reset(self):
        self.offset = 0


def yolo_head(feats, anchors):
        """Convert final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        grid_size = tf.shape(feats)[1]  # height, width
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=feats.dtype), [1, 1, 1, num_anchors, 2])

        # Reshape to (batch, height, width, num_anchors, box_params)
        feats = tf.reshape(feats, [-1, grid_size, grid_size, num_anchors, 5])

        # Grid
        grid_x = tf.range(grid_size, dtype=feats.dtype)
        grid_y = tf.range(grid_size, dtype=feats.dtype)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        grid = tf.stack([grid_x, grid_y], axis=-1)
        grid = tf.expand_dims(grid, axis=2)
        grid = tf.expand_dims(grid, axis=0)

        # Adjust predictions to each spatial grid point and anchor size
        box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_size, feats.dtype)
        box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor
        box_confidence = tf.sigmoid(feats[..., 4:])

        return box_xy, box_wh, box_confidence


def freeze_yolov3_layers(model, freeze_until_layer):
    """
    Freezes the layers in the YOLOv3 model up to the specified layer.

    Parameters:
    model (tf.keras.Model): The YOLOv3 model.
    freeze_until_layer (str or int): The name or index of the layer until which to freeze.

    Returns:
    tf.keras.Model: The YOLOv3 model with specified layers frozen.
    """
    # If freeze_until_layer is an integer, freeze layers by index
    if isinstance(freeze_until_layer, int):
        for i, layer in enumerate(model.layers):
            if i <= freeze_until_layer:
                layer.trainable = False
            else:
                layer.trainable = True
    # If freeze_until_layer is a string, freeze layers by name
    elif isinstance(freeze_until_layer, str):
        freeze = True
        for layer in model.layers:
            if freeze:
                layer.trainable = False
                if layer.name == freeze_until_layer:
                    freeze = False
            else:
                layer.trainable = True
    else:
        raise ValueError("freeze_until_layer must be an integer or string")

    return model


def process_outputs(outputs, anchors, confidence_thresh=0.5, iou_thresh=0.5):
    boxes = []
    confidences = []

    for i in range(len(outputs)):
        box_xy, box_wh, box_confidence = yolo_head(feats = outputs[i], anchors = anchors[i])
        bx = tf.reshape(box_xy[..., 0], [-1]).numpy()
        by = tf.reshape(box_xy[..., 1], [-1]).numpy()
        bw = tf.reshape(box_wh[..., 0], [-1]).numpy()
        bh = tf.reshape(box_wh[..., 1], [-1]).numpy()
        conf = tf.reshape(box_confidence, [-1]).numpy()

        # Convert bounding box coordinates to top-left, bottom-right format
        boxes.append(np.stack([bx - (bw / 2), by - (bh / 2), bw, bh], axis=-1))
        confidences.append(conf)

    boxes = np.concatenate(boxes, axis=0)
    confidences = np.concatenate(confidences, axis=0)

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), confidence_thresh, iou_thresh)

    if len(indices) > 0:
        indices = indices.flatten()
        boxes = boxes[indices]
        confidences = confidences[indices]

    return boxes, confidences

def draw_boxes(image, boxes, confidences, confidence_threshold =0.5):
    for box, confidence in zip(boxes, confidences):
        if confidence < confidence_threshold:
              continue
        x_min, y_min, w, h = box
        x_max = (x_min + w)*image.shape[1]
        y_max = (y_min + h)*image.shape[0]
        cv2.rectangle(image, (int(x_min*image.shape[1]), int(y_min*image.shape[0])), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(image, f'{confidence:.2f}', (int(x_min*image.shape[1]), int(y_min*image.shape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


class trueBoundingBoxTransformer:
    def __init__(self, anchors, target_sizes=[13, 26, 52]):
        self.anchors = anchors
        self.target_sizes = target_sizes

    def calculate_iou(self, box1, box2):
        w1, h1 = box1
        w2, h2 = box2

        inter_width = min(w1, w2)
        inter_height = min(h1, h2)
        inter_area = inter_width * inter_height

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    def transform(self, data):
        anchor_size = len(self.anchors)
        grids = {}
        for r, size in enumerate(self.target_sizes):
            grids[r] = np.zeros((data.shape[0], size, size, anchor_size, 5))
        for m in range(data.shape[0]):
            for bbox in data.iloc[m, 1]:
                k = np.argmax([self.calculate_iou(bbox[2:], anchor) for anchor in self.anchors])
                for r, size in enumerate(self.target_sizes):
                    j = int(bbox[0] * size)
                    i = int(bbox[1] * size)
                    grids[r][m, i, j, k] = np.append(bbox, 1)
        return grids[0], grids[1], grids[2]


class YOLOv3Metrics:
    def __init__(self, anchors, confidence_thresh=0.5, iou_thresh=0.5, NMS_thresh = 0.0):
        self.anchors = anchors
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.NMS_thresh = NMS_thresh

    @staticmethod
    def cal_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Coordinates of the intersection rectangle
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        # Area of the intersection rectangle
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Area of both the prediction and ground-truth rectangles
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Union area
        union_area = box1_area + box2_area - inter_area

        # IoU calculation
        iou = inter_area / union_area if union_area != 0 else 0
        return iou

    def yolo_head(self, feats, anchors):
        """Convert final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        grid_size = tf.shape(feats)[1]  # height, width
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=feats.dtype), [1, 1, 1, num_anchors, 2])

        # Reshape to (batch, height, width, num_anchors, box_params)
        feats = tf.reshape(feats, [-1, grid_size, grid_size, num_anchors, 5])

        # Grid
        grid_x = tf.range(grid_size, dtype=feats.dtype)
        grid_y = tf.range(grid_size, dtype=feats.dtype)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        grid = tf.stack([grid_x, grid_y], axis=-1)
        grid = tf.expand_dims(grid, axis=2)
        grid = tf.expand_dims(grid, axis=0)

        # Adjust predictions to each spatial grid point and anchor size
        box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_size, feats.dtype)
        box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor
        box_confidence = tf.sigmoid(feats[..., 4:])

        return box_xy, box_wh, box_confidence

    def metrics(self, outputs, true_y):
        boxes = []
        true_boxes = []
        confidences = []
        true_confidences = []

        true_box_xy = true_y[..., 0:2]  # (x, y)
        true_box_wh = true_y[..., 2:4]  # (w, h)
        true_conf = true_y[..., 4:]  # objectness score
        
        tx = tf.reshape(true_box_xy[..., 0], [-1]).numpy()
        ty = tf.reshape(true_box_xy[..., 1], [-1]).numpy()
        tw = tf.reshape(true_box_wh[..., 0], [-1]).numpy()
        th = tf.reshape(true_box_wh[..., 1], [-1]).numpy()
        true_conf = tf.reshape(true_conf, [-1]).numpy()

        # Convert bounding box coordinates to top-left
        true_boxes.append(np.stack([tx - (tw / 2), ty - (th / 2), tw, th], axis=-1))
        true_confidences.append(true_conf)

        for i in range(len(outputs)):
            box_xy, box_wh, box_confidence = self.yolo_head(outputs[i], self.anchors[i])

            bx = tf.reshape(box_xy[..., 0], [-1]).numpy()
            by = tf.reshape(box_xy[..., 1], [-1]).numpy()
            bw = tf.reshape(box_wh[..., 0], [-1]).numpy()
            bh = tf.reshape(box_wh[..., 1], [-1]).numpy()
            conf = tf.reshape(box_confidence, [-1]).numpy()

            # Convert bounding box coordinates to top-left
            boxes.append(np.stack([bx - (bw / 2), by - (bh / 2), bw, bh], axis=-1))
            confidences.append(conf)

        boxes = np.concatenate(boxes, axis=0)
        true_boxes = np.concatenate(true_boxes, axis=0)
        confidences = np.concatenate(confidences, axis=0)
        true_confidences = np.concatenate(true_confidences, axis=0)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.confidence_thresh, self.NMS_thresh)
        true_boxes = true_boxes[true_confidences >= self.confidence_thresh]
        tp = 0
        fp = 0
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = boxes[indices]

            # Calculate metrics
            for pred_box in boxes:
                match_found = False
                for true_box in true_boxes:
                    iou = self.cal_iou(pred_box, true_box)
                    if iou >= self.iou_thresh:
                        tp += 1
                        match_found = True
                        break
                if not match_found:
                    fp += 1
        fn = len(true_boxes) - tp
        return tp, fp, fn
    


def crop_and_resize(image, number_plate_bbox, output_size=(416, 416), zoom_range=(1.0, 4.0), thresh_area= (0.01, 0.1)):
    """
    Crops around the number plate bounding box, resizes to the specified output size, and transforms the bounding box coordinates.

    Parameters:
    - image: Input image as a NumPy array.
    - number_plate_bbox: Normalized bounding box coordinates [x_center, y_center, width, height] of the number plate.
                         Coordinates are normalized between 0 and 1.
    - output_size: Desired output size (width, height) for the resized image.
    - zoom_factor: Factor by which to zoom into the number plate area.
    - thresh_area: Threshold area (normalized) to determine if zoom should be applied or not.

    Returns:
    - resized_image: The cropped and resized image, or None if the bounding box area is below the threshold.
    - transformed_bbox: Transformed bounding box coordinates [x_center, y_center, width, height] in the resized image, or None if the bounding box area is below the threshold.
    - valid: A boolean flag indicating whether the image and bounding box are valid (True) or should be skipped (False).
    """
    x_center, y_center, w, h = number_plate_bbox

    # Convert normalized coordinates to pixel coordinates
    image_height, image_width, _ = image.shape
    x_center = x_center * image_width
    y_center = y_center * image_height
    w = w * image_width
    h = h * image_height

    # Calculate bounding box area (normalized)
    bbox_area = (w * h) / (image_width * image_height)

    # If bounding box area is below threshold, return None and valid flag as False
    if bbox_area < thresh_area[0] or bbox_area > thresh_area[1]:
        return None, None, False

    # Calculate top-left corner of the bounding box
    x = int(max(0, x_center - w / 2))
    y = int(max(0, y_center - h / 2))


    # Calculate the crop size based on the maximum of width (w)
    crop_size = max(w, h)

    # Choose a random zoom factor within the specified range
    zoom_factor = random.uniform(zoom_range[0], zoom_range[1])

    # Calculate zoomed-in bounding box
    zoomed_x = int(max(0, x - (zoom_factor - 1) * crop_size / 2))
    zoomed_y = int(max(0, y - (zoom_factor - 1) * crop_size / 2))
    zoomed_w = int(min(image_width - zoomed_x, crop_size * zoom_factor))
    zoomed_h = int(min(image_height - zoomed_y, crop_size * zoom_factor))

    # Crop the image
    cropped_image = image[zoomed_y:zoomed_y + zoomed_h, zoomed_x:zoomed_x + zoomed_w]

    # Resize the image to the output size
    resized_image = cv2.resize(cropped_image, output_size)

    # Calculate the transformation ratio
    resize_ratio_w = output_size[0] / zoomed_w
    resize_ratio_h = output_size[1] / zoomed_h

    # Transform the bounding box coordinates
    transformed_x_center = (x_center - zoomed_x) * resize_ratio_w
    transformed_y_center = (y_center - zoomed_y) * resize_ratio_h
    transformed_w = w * resize_ratio_w
    transformed_h = h * resize_ratio_h

    # Normalize the transformed bounding box coordinates
    transformed_x_center /= output_size[0]
    transformed_y_center /= output_size[1]
    transformed_w /= output_size[0]
    transformed_h /= output_size[1]

    transformed_bbox = [transformed_x_center, transformed_y_center, transformed_w, transformed_h]

    return resized_image, transformed_bbox, True

def bbox_to_center_and_size(x_min, y_min, x_max, y_max):
    width =  int(x_max - x_min)
    height = int(y_max - y_min)
    center_x = int(round(x_min + width / 2))
    center_y = int(round(y_min + height / 2))
    return [center_x, center_y, width, height]

def img_bbox_resize(img_path, bboxes, targetsize = 416):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    y = img.shape[0]
    x = img.shape[1]
    img = cv2.resize(img, (targetsize, targetsize))

    out_bbox = []
    for bbox in bboxes:
        center_x, center_y, width, height = bbox

        center_x = center_x / x
        center_y = center_y / y
        width = width / x
        height = height / y

        out_bbox.append([center_x, center_y, width, height])

    return img, out_bbox

def calculate_iou(box1, box2):
    w1, h1 = box1
    w2, h2 = box2

    inter_width = min(w1, w2)
    inter_height = min(h1, h2)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def multi_output_image_generator(df, y_13, y_26, y_52, batch_size):
    num_samples = len(df)
    indices = np.arange(num_samples)

    while True:  # Infinite loop
        np.random.shuffle(indices)  # Shuffle indices at the start of each epoch
        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset + batch_size]
            batch_samples = df.iloc[batch_indices]

            # Assuming images are already preprocessed and stored in 'img' column of df
            images = np.stack(batch_samples['img'].values)  # Assuming 'img' contains preprocessed images

            # Assuming y_13, y_26, y_52 are numpy arrays matching batch_size
            batch_y_13 = y_13[batch_indices]
            batch_y_26 = y_26[batch_indices]
            batch_y_52 = y_52[batch_indices]

            yield images, (batch_y_13, batch_y_26, batch_y_52)