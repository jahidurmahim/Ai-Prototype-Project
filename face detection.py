# Detects faces using Haar cascades and estimates distance of the nearest face based on known face width.

import cv2

class FaceDetector:
    # Initializes the face detector with required parameters and default values
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.KNOWN_DISTANCE = 50
        self.KNOWN_WIDTH = 16
        self.KNOWN_PIXELS = 250
        self.face_count = 0
        self.current_distance = float('0')
        self.faces = []
        self.frame_count = 0

    # Calculates the distance from the camera based on face width in pixels
    def calculate_distance(self, face_width_pixels):
        return (self.KNOWN_WIDTH * self.KNOWN_PIXELS) / face_width_pixels if face_width_pixels > 0 else float('0')

    # Detects faces in the given frame and calculates distance for the first face
    def detect(self, frame):
        self.frame_count += 1

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        self.face_count = len(self.faces)
        self.current_distance = float('0')

        if self.face_count > 0:
            x, y, w, h = self.faces[0]  # Use first face for distance
            self.current_distance = self.calculate_distance(w)

        return display_frame

    # Returns the current detection status including face count and distance
    def get_status(self):
        return {'face_count': self.face_count, 'distance': self.current_distance, 'faces': self.faces}

    # Placeholder method (not used currently)
    def close(self):
        pass
