# Real-time face spoof detection system using blur, texture, color, and reflection analysis

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import entropy

class SpoofDetectionSystem:
    def __init__(self, cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml', history_size=12, spoof_threshold=0.40):
        # Initialize face detector and parameters for spoof detection history and threshold
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")
        self.history_size = history_size
        self.spoof_history = []
        self.spoof_threshold = spoof_threshold

    def detect_blur(self, face_roi):
        # Detect blur in the face region using variance of Laplacian operator
        if face_roi.size == 0:
            return False, 0.0
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_value < 100, blur_value

    def detect_texture_variation(self, face_roi):
        # Analyze texture variation using Local Binary Pattern entropy
        if face_roi.size == 0:
            return False, 0.0
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, P=n_points, R=radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2), density=True)
        lbp_entropy = entropy(hist)
        return lbp_entropy < 2.5, lbp_entropy

    def detect_color_variance(self, face_roi):
        # Detect unnatural color variance in the face region
        if face_roi.size == 0:
            return False, 0.0
        face_roi = cv2.resize(face_roi, (100, 100))
        b, g, r = cv2.split(face_roi)
        b_var = np.var(b)
        g_var = np.var(g)
        r_var = np.var(r)
        cross_var = np.var([np.mean(b), np.mean(g), np.mean(r)])
        avg_color_var = (b_var + g_var + r_var) / 3
        is_spoof = avg_color_var < 500 or cross_var < 100
        return is_spoof, avg_color_var

    def detect_reflection(self, face_roi):
        # Detect unusual reflections or lighting patterns indicating spoofing
        if face_roi.size == 0:
            return False, 0.0
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        hist, _ = np.histogram(gradient_magnitude, bins=50)
        hist = hist / np.sum(hist)
        peak = np.max(hist)
        mean = np.mean(hist)
        peak_to_mean = peak / mean if mean > 0 else 0
        return peak_to_mean > 7, peak_to_mean

    def detect_spoof(self, frame, face_box=None):
        # Combine multiple spoof detection methods to classify a face region as spoof or real
        if face_box is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            if len(faces) == 0:
                return "No Face Detected"
            face_box = faces[0]

        if len(face_box) != 4:
            return "Invalid Face Box"

        x, y, w, h = face_box
        face_roi = frame[y:y + h, x:x + w]
        if face_roi.size == 0:
            return "No Face Detected"

        is_blurry, _ = self.detect_blur(face_roi)
        low_texture, _ = self.detect_texture_variation(face_roi)
        bad_color, _ = self.detect_color_variance(face_roi)
        bad_reflection, _ = self.detect_reflection(face_roi)

        spoof_score = (0.25 * float(is_blurry) +
                       0.25 * float(low_texture) +
                       0.3 * float(bad_color) +
                       0.2 * float(bad_reflection))

        self.spoof_history.append(spoof_score > 0.5)
        if len(self.spoof_history) > self.history_size:
            self.spoof_history.pop(0)

        spoof_ratio = sum(self.spoof_history) / max(len(self.spoof_history), 1)
        return "Spoof Detected" if spoof_ratio > self.spoof_threshold else "Real Face"

    def process_frame(self, frame):
        # Detect faces in the frame and perform spoof detection with visual feedback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            result = self.detect_spoof(frame, (x, y, w, h))
            color = (0, 0, 255) if result == "Spoof Detected" else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, result, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

