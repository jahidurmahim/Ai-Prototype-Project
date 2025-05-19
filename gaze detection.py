# Detects head orientation using MediaPipe FaceMesh and determines if the user's head is centered based on yaw rotation.

import cv2
import mediapipe as mp
import numpy as np

class GazeDetector:
    # Initializes the gaze detector, setting up MediaPipe FaceMesh and camera parameters
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            refine_landmarks=True
        )
        # Landmark indices
        self.NOSE_TIP, self.LEFT_TEMPLE, self.RIGHT_TEMPLE = 1, 234, 454
        # Camera parameters
        self.camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        # Threshold for head rotation
        self.rotation_threshold = 30  # Degrees

    # Calculates head rotation angles (pitch, yaw, roll) from facial landmarks
    def get_face_rotation(self, landmarks):
        model_points = np.array([
            [0.0, 0.0, 0.0],        # Nose tip
            [0.0, -330.0, -65.0],   # Chin
            [-225.0, 170.0, -135.0],# Left temple
            [225.0, 170.0, -135.0], # Right temple
            [-150.0, 75.0, -125.0], # Left eye corner
            [150.0, 75.0, -125.0]   # Right eye corner
        ], dtype=np.float32)
        image_points = np.array([
            [landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y],
            [landmarks[152].x, landmarks[152].y],  # Chin
            [landmarks[self.LEFT_TEMPLE].x, landmarks[self.LEFT_TEMPLE].y],
            [landmarks[self.RIGHT_TEMPLE].x, landmarks[self.RIGHT_TEMPLE].y],
            [landmarks[362].x, landmarks[362].y],  # Left eye corner
            [landmarks[33].x, landmarks[33].y]     # Right eye corner
        ], dtype=np.float32)
        image_points[:, 0] *= 1280  # Scale to image width
        image_points[:, 1] *= 720   # Scale to image height
        success, rotation_vec, _ = cv2.solvePnP(model_points, image_points, self.camera_matrix, self.dist_coeffs)
        if not success:
            return [0, 0, 0]
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
        yaw = np.arctan2(-rotation_mat[2, 0], np.sqrt(rotation_mat[2, 1] ** 2 + rotation_mat[2, 2] ** 2))
        roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        return np.degrees([pitch, yaw, roll])

    # Detects if head is centered based on rotation and return status
    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return "No Face Detected"

        landmarks = results.multi_face_landmarks[0].landmark
        head_pose = self.get_face_rotation(landmarks)
        yaw = head_pose[1]  # Use yaw for left-right rotation

        # Check if head is centered
        status = "Centered" if abs(yaw) <= self.rotation_threshold else "Not Centered"

        return status

    # Releases resources used by MediaPipe FaceMesh
    def close(self):
        self.face_mesh.close()
