import cv2
import os
import numpy as np
import tensorflow as tf
from simple_tracker import SimpleTracker
from tensorflow.nn import softmax
from glob import glob
import matplotlib.pyplot as plt

from keras.saving import register_keras_serializable
@register_keras_serializable()
def split_bbox(t):
    return t[:, :, :4]

@register_keras_serializable()
def split_objectness(t):
    return tf.sigmoid(t[:, :, 4])

@register_keras_serializable()
def split_class_logits(t):
    return t[:, :, 5:]

# --- Paths ---
FRAME_DIR = "../data/VisDrone2019/val/sequences/uav0000137_00458_v"
frame_paths = sorted(glob(os.path.join(FRAME_DIR, "*.jpg")))

# --- Load Model & Tracker ---
model = tf.keras.models.load_model("custom_model.keras")
tracker = SimpleTracker(max_lost=5, iou_threshold=0.3)

label_map = {0: "background", 1: "pedestrian", 2: "car", 3: "truck", 4: "bus"}  # update to match your dataset
IMG_SIZE = (224, 224)

# --- Process Each Frame ---
for img_path in frame_paths:
    print(f"[INFO] Processing {img_path}")
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[WARNING] Couldn't load image: {img_path}")
        continue

    orig = frame.copy()
    h, w, _ = orig.shape

    # Preprocess
    input_img = cv2.resize(orig, IMG_SIZE) / 255.0
    input_tensor = tf.expand_dims(input_img, axis=0)

    # Inference
    bbox_pred, objectness_pred, class_logits = model.predict(input_tensor)
    objectness_pred = tf.sigmoid(objectness_pred[0])
    bbox_pred = bbox_pred[0]
    class_probs = tf.nn.softmax(class_logits[0], axis=-1)
    class_ids = tf.argmax(class_probs, axis=-1).numpy()
    scores = objectness_pred.numpy()

    # Filter and collect boxes
    threshold = 0.3
    boxes = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            box = bbox_pred[i]
            box = [
                int(box[0] * w / IMG_SIZE[1]),
                int(box[1] * h / IMG_SIZE[0]),
                int(box[2] * w / IMG_SIZE[1]),
                int(box[3] * h / IMG_SIZE[0])
            ]
            boxes.append((box, int(class_ids[i])))

    print(f"[DEBUG] Detections: {len(boxes)}")

    # Tracker
    tracks = tracker.update(boxes)

    for track_id, data in tracks.items():
        x1, y1, x2, y2 = map(int, data["box"])
        cls = data["class"]
        label = label_map.get(cls, str(cls))
        text = f"ID:{track_id} {label}"
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Show
    cv2.imshow("Tracking", orig)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
