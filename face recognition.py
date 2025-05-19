# Performs real-time face recognition using Haar cascades and ORB feature matching with confidence scoring and face locking.

import cv2
import os
import numpy as np

class FaceRecognitionSystem:
    # Initialize face detector, ORB feature extractor, and variables for known faces and locking mechanism
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}  # Dictionary to store {name: (keypoints, descriptors)}
        self.locked_name = None  # The name to lock after first detection
        self.orb = cv2.ORB_create(nfeatures=400)  # Increased features for more detail
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Preprocess face region for improved feature extraction and matching
    def preprocess_face(self, face_roi):
        """Preprocess face image for better matching with enhanced contrast and normalization."""
        face_roi = cv2.resize(face_roi, (120, 120))  # Resize to standard size
        face_roi = cv2.equalizeHist(face_roi)  # Histogram equalization for lighting invariance
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  # Contrast adjustment
        face_roi = clahe.apply(face_roi)
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0.5)  # Noise reduction
        face_roi = cv2.normalize(face_roi, None, 0, 255, cv2.NORM_MINMAX)  # Normalize intensity
        return face_roi

    # Load known faces from a directory and extract ORB features for each face
    def load_known_faces(self, directory_path):
        """Load known faces from directory using ORB features."""
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return

        for image_name in os.listdir(directory_path):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(image_name)[0]
                image_path = os.path.join(directory_path, image_name)

                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Could not read image: {image_path}")
                        continue

                    faces = self.face_cascade.detectMultiScale(img, 1.1, 4, minSize=(90, 90))  # Detect faces
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = self.preprocess_face(img[y:y + h, x:x + w])
                        keypoints, descriptors = self.orb.detectAndCompute(face_roi, None)
                        if descriptors is not None and len(keypoints) > 15:  # Validate feature count
                            self.known_faces[name] = (keypoints, descriptors)
                            print(f"Loaded face: {name} with {len(keypoints)} keypoints")
                        else:
                            print(f"Insufficient features for: {image_name}")
                    else:
                        print(f"No face found in: {image_name}")
                except Exception as e:
                    print(f"Error with {image_name}: {e}")

        print(f"Loaded {len(self.known_faces)} faces")

    # Match a detected face against known faces using ORB feature matching
    def match_face(self, face_roi):
        """Match face against templates using ORB feature matching."""
        best_match = "Unknown"
        best_score = 0.0  # Initialize minimum match score

        keypoints, descriptors = self.orb.detectAndCompute(face_roi, None)
        if descriptors is None or len(keypoints) < 15:
            return best_match, best_score

        for name, (template_kp, template_desc) in self.known_faces.items():
            matches = self.bf.match(descriptors, template_desc)
            if len(matches) > 8:
                distances = [match.distance for match in matches]
                avg_distance = np.mean(distances)
                max_distance = 80.0  # ORB max distance threshold
                similarity = 1.0 - (avg_distance / max_distance) if avg_distance < max_distance else 0.0
                if similarity > best_score and similarity > 0.35:  # Threshold for recognition
                    best_score = similarity
                    best_match = name

        return best_match, best_score

    # Detect faces in frame and return recognized name, using locking to keep first recognized face
    def detect(self, frame):
        """Detect and identify faces in the frame, returning only the recognized name."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(90, 90))

        if len(faces) == 0:
            self.locked_name = None
            return "No Face Detected"

        x, y, w, h = faces[0]
        aspect_ratio = w / float(h)
        if not (0.75 < aspect_ratio < 1.3):
            self.locked_name = None
            return "No Face Detected"

        if self.locked_name is not None:
            return self.locked_name

        face_roi = self.preprocess_face(gray[y:y + h, x:x + w])
        current_name, _ = self.match_face(face_roi)

        if current_name != "Unknown":
            self.locked_name = current_name
        return current_name

    # Detect faces and return both recognized name and confidence score
    def detect_with_confidence(self, frame):
        """Detect and identify faces in the frame, returning a tuple of (recognized name, confidence)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(90, 90))

        if len(faces) == 0:
            return "No Face Detected", 0.0
        x, y, w, h = faces[0]
        aspect_ratio = w / float(h)
        if not (0.75 < aspect_ratio < 1.3):
            return "No Face Detected", 0.0

        face_roi = self.preprocess_face(gray[y:y + h, x:x + w])
        current_name, confidence = self.match_face(face_roi)
        return current_name, confidence
