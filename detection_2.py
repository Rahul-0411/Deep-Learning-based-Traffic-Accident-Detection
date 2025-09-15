import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Layer
from ultralytics import YOLO

class Cast(Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)
    
yolo_model = YOLO("E:/Final_BE_Project/work/test/yolov5lu.pt")
cnn_model = load_model("E:/Final_BE_Project/work/test/roiresnetv2.h5")


vehicle_classes = ["car", "truck", "bus", "motorcycle"]
obstacle_classes = ["traffic light", "stop sign", "fire hydrant"]

def are_boxes_close(box1, box2, img_w, img_h, rel_thresh=0.05):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
    cx2, cy2 = (x1b + x2b) / 2, (y1b + y2b) / 2
    dist = math.hypot(cx1 - cx2, cy1 - cy2)
    avg_size = ((x2 - x1 + y2 - y1) + (x2b - x1b + y2b - y1b)) / 4
    return dist / avg_size < 2.0, dist / avg_size

def combine_boxes(boxes, padding=0.1):
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    w, h = x2 - x1, y2 - y1
    return (max(0, x1 - int(w*padding)), max(0, y1 - int(h*padding)),
            x2 + int(w*padding), y2 + int(h*padding))

def detect_irregular_orientation(box, std_ratio=1.5):
    x1, y1, x2, y2 = box
    ar = (x2 - x1) / max((y2 - y1), 1)
    return abs(ar - std_ratio) > 0.5

def make_square_roi(x1, y1, x2, y2, img_w, img_h):
    w, h = x2 - x1, y2 - y1
    if w > h:
        diff = w - h
        y1 = max(0, y1 - diff // 2)
        y2 = min(img_h, y2 + diff // 2)
    else:
        diff = h - w
        x1 = max(0, x1 - diff // 2)
        x2 = min(img_w, x2 + diff // 2)
    return x1, y1, x2, y2

def preprocess_roi(roi):
    roi = cv2.resize(roi, (224, 224))
    roi = roi.astype('float32') / 255.0
    return roi

frame_id, skip_rate = 0, 5
CONF_THRESHOLD = 0.6

def process_frame(frame):

    img = frame.copy()
    h, w, _ = img.shape
    results = yolo_model(img)

    vehicle_boxes, obstacle_boxes = [], []
    for r in results:
        for box in r.boxes:
            cid = int(box.cls[0])
            name = yolo_model.names[cid]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if name in vehicle_classes and conf > 0.6:
                vehicle_boxes.append((x1, y1, x2, y2, name, conf))
            elif name in obstacle_classes and conf > 0.6:
                obstacle_boxes.append((x1, y1, x2, y2, name, conf))
    if not vehicle_boxes:
        return []
    
    unusual_ids = []
    for idx, (x1, y1, x2, y2, _, _) in enumerate(vehicle_boxes):
        if detect_irregular_orientation((x1, y1, x2, y2)):
            unusual_ids.append(idx)

    accident_rois, processed, square_coords, batch_input = [], set(), [], []

    for i, box1 in enumerate(vehicle_boxes):
        if i in processed:
            continue
        x1, y1, x2, y2, _, _ = box1
        close_ids, indicators = [i], 2 if i in unusual_ids else 0
        boxes = [box1[:4]]

        for j, box2 in enumerate(vehicle_boxes[i+1:], start=i+1):
            if j in processed:
                continue
            close, _ = are_boxes_close(box1[:4], box2[:4], w, h)
            if close:
                close_ids.append(j)
                boxes.append(box2[:4])
                indicators += 2

        if indicators >= 2:
            mx1, my1, mx2, my2 = combine_boxes(boxes)
            sx1, sy1, sx2, sy2 = make_square_roi(mx1, my1, mx2, my2, w, h)
            if sx2 - sx1 < 50 or sy2 - sy1 < 50:
                continue
            roi = img[sy1:sy2, sx1:sx2]
            batch_input.append(preprocess_roi(roi))
            square_coords.append((sx1, sy1, sx2, sy2))
            processed.update(close_ids)
    
    if batch_input:
            batch_input = np.array(batch_input)
            preds = cnn_model.predict(batch_input)
            results_list = []
            for (x1, y1, x2, y2), pred in zip(square_coords, preds):
                score = float(pred[0])  # Assuming CNN outputs probability of "Non-Accident"
                label = "Non-Accident" if score > CONF_THRESHOLD else "Accident"
                results_list.append((x1, y1, x2, y2, label, score))
            return results_list
    else:
        return []