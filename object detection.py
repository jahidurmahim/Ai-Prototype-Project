# Detects mobile phones or laptops in video frames using YOLOv3-tiny model

import cv2
import numpy as np
import os

class PhoneDetector:
    def __init__(self):
        self.net = None
        self.classes = None

        # Get current directory path to load model files correctly
        base_path = os.path.dirname(os.path.abspath(__file__))

        # Paths to YOLO tiny weights, config, and class names files
        weights_path = os.path.join(base_path, "yolov3-tiny.weights")
        config_path = os.path.join(base_path, "yolov3-tiny.cfg")
        names_path = os.path.join(base_path, "coco.names")

        # Load YOLO model and classes if all files exist
        if os.path.exists(weights_path) and os.path.exists(config_path) and os.path.exists(names_path):
            self.net = cv2.dnn.readNet(weights_path, config_path)
            with open(names_path, "r") as f:
                self.classes = f.read().strip().split("\n")

    # Process a frame to detect phones or laptops using YOLO
    def process_frame(self, frame):
        if self.net is None:
            return "YOLO not initialized"  # Model not loaded, cannot detect

        # Prepare input blob for YOLO: resize, normalize, and reorder channels
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416, 416)),
                                     1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Run forward pass to get output from YOLO detection layers
        outputs = self.net.forward([self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()])

        detected = False
        # Iterate over all detections in the output layers
        for output in outputs:
            for detection in output:
                scores = detection[5:]  # Class confidence scores start at index 5
                class_id = np.argmax(scores)  # Get index of highest score

                # Check if confidence is above threshold and class is cell phone or laptop
                if scores[class_id] > 0.5 and self.classes[class_id] in ["cell phone", "laptop"]:
                    detected = True
                    break  # Exit once an object is detected
            if detected:
                break

        # Return detection result as a string
        return "Object detected" if detected else "No object detected"
