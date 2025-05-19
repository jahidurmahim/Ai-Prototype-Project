# This main function combines all features and runs them together as one program

import cv2
import numpy as np
import threading
import time
import os
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from idextract import IDExtractor
from face_detection import FaceDetector
from gaze_detection import GazeDetector
from face_recognize import FaceRecognitionSystem
from objectdetectmain import PhoneDetector
from face_spoofying import SpoofDetectionSystem

class CheatDetectionAI:
    def __init__(self):
        # Initialize camera with robust error handling
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            raise RuntimeError("Failed to initialize camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.id_extractor = IDExtractor()
        self.face_detector = FaceDetector()
        self.gaze_detector = GazeDetector()
        self.gaze_detector.face_detector = self.face_detector
        self.face_recognizer = FaceRecognitionSystem()
        self.face_recognizer.load_known_faces('known_faces')
        self.phone_detector = PhoneDetector()
        if self.phone_detector.net is None:
            print("YOLO files missing! Object detection disabled.")
        self.spoof_detector = SpoofDetectionSystem()
        self.results = {"ID Extraction": None, "Face Detection": None, "Face Distance": None, "Gaze Detection": None,
                        "Face Recognition": None, "Object Detection": None, "Spoof Detection": None}
        self.alert_active = False
        self.start_time = None
        self.blink_state = True
        self.frame_count = 0
        self.last_alert_time = time.time()
        self.last_blink_time = time.time()
        self.welcome_start_time = None
        self.lock = threading.Lock()
        self.setup_google_sheets()
        self.cheat_incidents = []
        self.cheat_count = 0
        self.interviewee_serial = self.get_next_serial_number()
        self.extracted_id = None
        self.matched_name = None
        self.verified_name = None
        self.active_incidents = {}
        self.gaze_not_centered_start = None

    def setup_google_sheets(self):
        try:
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name('cheatdetectionai-3172af5828f2.json', scope)
            self.client = gspread.authorize(creds)
            self.spreadsheet = self.client.open_by_key('1fnaY-4V0lNdG10MKb2tU4fLo4V54fC1oIadojohY9kU')
            self.sheet = self.spreadsheet.get_worksheet(0)
            headers = ["Interviewee Serial No", "Extracted ID", "Name", "Cheat Count", "Cheat Details", "Cheat Detection Timings"]
            first_row = self.sheet.row_values(1)
            if not first_row or first_row != headers:
                self.sheet.update('A1:F1', [headers])
            print("Google Sheets connected successfully!")
        except Exception as e:
            print(f"Error connecting to Google Sheets: {e}")
            self.client = None
            self.sheet = None
            self.spreadsheet = None

    def get_next_serial_number(self):
        if not self.sheet:
            return 1
        try:
            serial_numbers = self.sheet.col_values(1)
            if serial_numbers and not serial_numbers[0].isdigit():
                serial_numbers = serial_numbers[1:]
            return 1 if not serial_numbers else int(serial_numbers[-1]) + 1
        except Exception as e:
            print(f"Error getting next serial number: {e}")
            return 1

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}.{secs:02d}"

    def log_cheat_incident(self, cheat_type):
        if cheat_type not in self.active_incidents:
            self.active_incidents[cheat_type] = time.time() - self.start_time if self.start_time else 0

    def end_cheat_incident(self, cheat_type):
        if cheat_type in self.active_incidents:
            start_time = self.active_incidents[cheat_type]
            end_time = time.time() - self.start_time if self.start_time else 0
            duration = end_time - start_time
            if duration >= 0:
                time_str = f"{self.format_time(start_time)} - {self.format_time(end_time)} mins"
                self.cheat_incidents.append({"type": cheat_type, "timestamp": time_str})
                self.cheat_count += 1
            del self.active_incidents[cheat_type]
            if cheat_type == "Gaze not centered":
                self.gaze_not_centered_start = None

    def save_to_sheets(self):
        if not self.sheet:
            print("Google Sheets not connected. Data will not be saved.")
            return False
        try:
            active_types = list(self.active_incidents.keys())
            for incident_type in active_types:
                self.end_cheat_incident(incident_type)
            cheat_details_list = []
            cheat_timings_list = []
            cheat_type_counts = {}
            for incident in self.cheat_incidents:
                formatted_type = self.format_cheat_type(incident['type'])
                cheat_type_counts[formatted_type] = cheat_type_counts.get(formatted_type, 0) + 1
                start_str, end_str = incident['timestamp'].split(' - ')
                cheat_timings_list.append(f"{start_str}-{end_str}")
            for cheat_type, count in cheat_type_counts.items():
                cheat_details_list.append(f"{cheat_type} ({count})")
            cheat_details = "\n".join(cheat_details_list)
            cheat_timings = "\n".join(cheat_timings_list)
            row_data = [self.interviewee_serial, self.extracted_id or "", self.verified_name or "", self.cheat_count, cheat_details, cheat_timings]
            self.sheet.append_row(row_data)
            print(f"Data saved to Google Sheets for interviewee #{self.interviewee_serial} with {self.cheat_count} cheat incidents")
            return True
        except Exception as e:
            print(f"Error saving to Google Sheets: {e}")
            return False

    def format_cheat_type(self, incident_type):
        type_map = {
            "No face detected": "Face Detection: No face detected",
            "Multiple faces detected": "Face Detection: Multiple faces detected",
            "Object detected": "Object Detection: Object detected",
            "Face too far": "Face Distance: Face too far",
            "Unknown face detected": "Face Recognition: Unknown face detected",
            "Gaze not centered": "Gaze Detection: Gaze not centered",
            "Face spoof detected": "Spoof Detection: Spoof detected"
        }
        return type_map.get(incident_type, f"- {incident_type}")

    def draw_ui(self, frame, showbetter=True, initial_phase=True, show_welcome=False, face_verification_phase=False, face_verified=False, face_not_verified=False, show_id_prompt=False, processing_face=False):
        if frame is None or frame.size == 0 or not frame.any():
            frame = np.zeros((720, 960, 3), dtype=np.uint8)
        video_display = cv2.resize(frame, (960, 720))
        result_display = np.zeros((720, 480, 3), dtype=np.uint8)
        title_text = "Cheat Detection AI"
        font_title = cv2.FONT_HERSHEY_COMPLEX
        font_scale_title = 0.7
        thickness_title = 2
        text_size_title = cv2.getTextSize(title_text, font_title, font_scale_title, thickness_title)[0]
        text_x_title = (480 - text_size_title[0]) // 2
        cv2.putText(result_display, title_text, (text_x_title, 30), font_title, font_scale_title, (255, 255, 255), thickness_title)
        if show_welcome:
            welcome_text = "Welcome To Cheat Detection AI System"
            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(welcome_text, font, font_scale, thickness)[0]
            text_x = (960 - text_size[0]) // 2
            text_y = (720 + text_size[1]) // 2
            cv2.putText(video_display, welcome_text, (text_x, text_y), font, font_scale, (200, 255, 200), thickness)
            combined = np.hstack((video_display, result_display))
            return combined
        if face_verification_phase:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(result_display, "Are you ready for face verification?", (10, 200), font, 0.7, (200, 255, 200), 2)
            if self.blink_state:
                cv2.putText(result_display, "Press 'y' to start verification or 'q' to stop", (10, 240), font, 0.6, (200, 200, 200), 1)
        elif processing_face:
            font = cv2.FONT_HERSHEY_COMPLEX
            if self.blink_state:
                cv2.putText(result_display, "Processing face verification...", (10, 200), font, 0.7, (200, 255, 200), 2)
        elif face_not_verified:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(result_display, "Verification Failed", (10, 100), font, 0.7, (0, 0, 255), 2)
            cv2.putText(result_display, "Face Not Recognized..", (10, 130), font, 0.7, (0, 0, 255), 2)
            if self.blink_state:
                cv2.putText(result_display, "Press 't' to try again or 'q' to stop", (10, 180), font, 0.6, (200, 200, 200), 1)
        elif face_verified:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(result_display, "Face Verified Successfully", (10, 120), font, 0.8, (0, 255, 0), 2)
            font_name = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result_display, f"Interviewee Name: {self.verified_name or 'Unknown'}", (20, 160), font_name, 0.6, (0, 255, 0), 2)
            id_text = "Show your ID card"
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(id_text, font, font_scale, thickness)[0]
            text_x = 10
            cv2.putText(result_display, id_text, (text_x, 270), font, font_scale, (255, 255, 250), thickness)
            if show_id_prompt and self.blink_state:
                cv2.putText(result_display, "Press 's' to extract ID or 'q' to stop", (10, 300), font, 0.6, (200, 200, 200), 1)
        elif initial_phase:
            font = cv2.FONT_HERSHEY_COMPLEX
            if self.blink_state:
                cv2.putText(result_display, "Show your ID card", (30, 200), font, 0.8, (0, 255, 0), 2)
                cv2.putText(result_display, "Press 's' for scanning", (30, 230), font, 0.8, (0, 255, 0), 2)
                cv2.putText(result_display, "Press 'q' to stop", (30, 260), font, 0.8, (0, 255, 0), 2)
        else:
            font = cv2.FONT_HERSHEY_COMPLEX
            if self.start_time:
                elapsed = int(time.time() - self.start_time)
                minutes, seconds = divmod(elapsed, 60)
                cv2.putText(result_display, f"Time: {minutes:02d}:{seconds:02d}", (10, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            y_pos = 120
            alert_conditions = []
            face_detected = self.results.get("Face Detection") == "1"
            for feature, result in self.results.items():
                if result is None or feature == "Face Recognition":
                    continue
                if feature == "ID Extraction":
                    color = (200, 255, 200)
                    if result and not result.endswith("(Y/N?)") and result != "Scanning..." and self.extracted_id:
                        color = (0, 255, 0)
                    cv2.putText(result_display, f"{feature}: {result}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_pos += 40
                    if result == "Scanning...":
                        if self.blink_state:
                            cv2.putText(result_display, "Press 's' to scan your ID", (10, y_pos), font, 0.6, (200, 200, 200), 1)
                        y_pos += 30
                    elif result and "(Y/N?)" in result:
                        if self.blink_state:
                            cv2.putText(result_display, "Press 'y' if the ID is correct", (10, y_pos), font, 0.6, (200, 200, 200), 1)
                            y_pos += 30
                            cv2.putText(result_display, "Press 'n' to rescan the ID", (10, y_pos), font, 0.6, (200, 200, 200), 1)
                        y_pos += 40
                    elif showbetter and self.verified_name and not result.endswith("(Y/N?)") and result != "Scanning...":
                        cv2.putText(result_display, f"Interviewee Name: {self.verified_name}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        y_pos += 40
                else:
                    is_red = False
                    if feature == "Face Detection" and result != "1":
                        is_red = True
                    elif feature == "Face Distance" and "cm" in str(result) and float(result.split()[0]) > 40:
                        is_red = True
                    elif feature == "Gaze Detection" and result != "Centered":
                        is_red = True
                    elif feature == "Object Detection" and result == "Object detected":
                        is_red = True
                    elif feature == "Spoof Detection" and result.lower() == "spoof detected":
                        is_red = True
                    alert_conditions.append(is_red)
                    color = (0, 0, 255) if is_red else (0, 255, 0)
                    cv2.putText(result_display, f"{feature}: {result}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    y_pos += 40
                    if feature == "Spoof Detection":
                        y_pos += 40
            self.alert_active = any(alert_conditions)
            if self.alert_active and self.blink_state:
                alert_text = "ALERT!!"
                font = cv2.FONT_HERSHEY_COMPLEX
                font_scale = 0.8
                thickness = 3
                text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
                text_x = (480 - text_size[0]) // 2
                text_y = (720 + text_size[1]) // 2 + 50
                cv2.putText(result_display, alert_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
        combined = np.hstack((video_display, result_display))
        return combined

    def run_id_extraction(self, frame, key):
        display_frame, id_result, id_verified = self.id_extractor.process_frame(frame, force_scan=(key == ord('s')), key_pressed=key)
        with self.lock:
            if id_verified:
                self.results["ID Extraction"] = id_result
                self.extracted_id = id_result
                display_frame = frame.copy()
                cv2.putText(display_frame, f"ID Verified: {id_result}", (30, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            elif id_result:
                self.results["ID Extraction"] = id_result + " (Y/N?)"
            else:
                self.results["ID Extraction"] = "Scanning..."
        return display_frame, id_verified

    def run_features(self, frame):
        threads = []
        prev_results = self.results.copy()
        def update_result(key, func):
            try:
                result = func()
                with self.lock:
                    self.results[key] = result
                    if key == "Face Recognition" and result[0] == "Unknown" and len(self.face_detector.faces) > 0:
                        for _ in range(20):
                            time.sleep(0.15)
                            retry_result, confidence = self.face_recognizer.detect_with_confidence(processed_frame)
                            if retry_result not in ["Unknown", "No Face Detected", "Multiple Faces Detected", "Face Too Small"]:
                                if confidence > result[1]:
                                    self.results[key] = (retry_result, confidence)
                                    result = (retry_result, confidence)
                                if confidence > 0.9:
                                    break
                    is_alert = False
                    alert_message = None
                    resolved = False
                    if key == "Face Detection":
                        if result != "1" and (prev_results.get(key) == "1" or prev_results.get(key) is None):
                            is_alert = True
                            alert_message = "No face detected" if result == "0" else "Multiple faces detected"
                        elif result == "1" and prev_results.get(key) != "1" and prev_results.get(key) is not None:
                            resolved = True
                            alert_message = "No face detected" if prev_results.get(key) == "0" else "Multiple faces detected"
                    elif key == "Face Distance":
                        if "cm" in str(result):
                            current_distance = float(result.split()[0])
                            if current_distance > 40:
                                if prev_results.get(key) is None or float(prev_results.get(key).split()[0]) <= 40:
                                    is_alert = True
                                    alert_message = "Face too far"
                                elif prev_results.get(key) is not None and float(prev_results.get(key).split()[0]) > 40:
                                    resolved = True
                                    alert_message = "Face too far"
                    elif key == "Gaze Detection":
                        if result != "Center":
                            current_time = time.time() - self.start_time if self.start_time else 0
                            if prev_results.get(key) == "Center" or prev_results.get(key) is None:
                                is_alert = True
                                alert_message = "Gaze not centered"
                                self.log_cheat_incident(alert_message)
                                self.gaze_not_centered_start = current_time
                        else:
                            if prev_results.get(key) != "Center" and prev_results.get(key) is not None:
                                resolved = True
                                alert_message = "Gaze not centered"
                                self.end_cheat_incident(alert_message)
                    elif key == "Face Recognition":
                        if result[0] == "Unknown" and (prev_results.get(key) is None or prev_results.get(key)[0] != "Unknown"):
                            is_alert = True
                            alert_message = "Unknown face detected"
                        elif result[0] != "Unknown" and prev_results.get(key) and prev_results.get(key)[0] == "Unknown":
                            resolved = True
                            alert_message = "Unknown face detected"
                            self.matched_name = result[0]
                    elif key == "Object Detection":
                        if result == "Object detected" and prev_results.get(key) != "Object detected":
                            is_alert = True
                            alert_message = "Object detected"
                        elif result != "Object detected" and prev_results.get(key) == "Object detected":
                            resolved = True
                            alert_message = "Object detected"
                    elif key == "Spoof Detection":
                        if result == "Spoof detected" and prev_results.get(key) != "Spoof detected":
                            is_alert = True
                            alert_message = "Face spoof detected"
                        elif result != "Spoof detected" and prev_results.get(key) == "Spoof detected":
                            resolved = True
                            alert_message = "Face spoof detected"
                    if self.start_time:
                        if is_alert and alert_message and alert_message not in self.active_incidents:
                            self.log_cheat_incident(alert_message)
                        elif resolved and alert_message and alert_message in self.active_incidents:
                            self.end_cheat_incident(alert_message)
                    self.alert_active = any((self.results[k] == "No Face Detected" or (k == "Face Detection" and self.results[k] != "1") or
                                          (k == "Face Distance" and float(self.results[k].split()[0]) > 40 if "cm" in self.results[k] else False) or
                                          (k == "Gaze Detection" and self.results[k] != "Center") or
                                          (k == "Face Recognition" and self.results[k][0] == "Unknown") or
                                          (k == "Object Detection" and self.results[k] == "Object detected") or
                                          (k == "Spoof Detection" and self.results[k] == "Spoof detected"))
                                          for k in self.results if self.results[k] is not None)
            except Exception as e:
                with self.lock:
                    self.results[key] = f"Error: {str(e)}"
        self.face_detector.detect(frame)
        processed_frame = frame.copy()
        if len(self.face_detector.faces) == 1:
            x, y, w, h = self.face_detector.faces[0]
            face_region = frame[y:y+h, x:x+w]
            if face_region.size > 0:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.equalizeHist(gray_face)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_face = clahe.apply(gray_face)
                gray_face = cv2.bilateralFilter(gray_face, 9, 75, 75)
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                if len(eyes) >= 2:
                    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = sorted(eyes, key=lambda e: e[0])[:2]
                    eye_center1 = (int(ex1 + ew1 // 2), int(ey1 + eh1 // 2))
                    eye_center2 = (int(ex2 + ew2 // 2), int(ey2 + eh2 // 2))
                    angle = np.arctan2(eye_center2[1] - eye_center1[1], eye_center2[0] - eye_center1[0]) * 180 / np.pi
                    center = (int(w // 2), int(h // 2))
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    gray_face = cv2.warpAffine(gray_face, M, (w, h))
                processed_frame[y:y+h, x:x+w] = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
        threads.append(threading.Thread(target=update_result, args=("Face Detection", lambda: str(self.face_detector.get_status()['face_count']))))
        threads.append(threading.Thread(target=update_result, args=("Face Distance", lambda: f"{self.face_detector.get_status()['distance']:.2f} cm")))
        threads.append(threading.Thread(target=update_result, args=("Gaze Detection", lambda: self.gaze_detector.detect(frame))))
        threads.append(threading.Thread(target=update_result, args=("Face Recognition", lambda: self.face_recognizer.detect_with_confidence(processed_frame))))
        if self.phone_detector.net is not None:
            threads.append(threading.Thread(target=update_result, args=("Object Detection", lambda: self.phone_detector.process_frame(frame))))
        else:
            self.results["Object Detection"] = "YOLO not initialized"
        threads.append(threading.Thread(target=update_result, args=("Spoof Detection", lambda: self.spoof_detector.detect_spoof(frame, self.face_detector.faces[0] if len(self.face_detector.faces) > 0 else None))))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        display_frame = frame.copy()
        face_count = int(self.results["Face Detection"] or 0)
        face_color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
        for (x, y, w, h) in self.face_detector.faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), face_color, 2)
            if self.matched_name and self.results["Face Recognition"][0] not in ["Unknown", "No Face Detected", "Multiple Faces Detected", "Face Too Small"]:
                cv2.putText(display_frame, self.matched_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, face_color, 2)
        if self.phone_detector.net and self.results["Object Detection"] == "Object detected":
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416, 416)), 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.phone_detector.net.setInput(blob)
            outputs = self.phone_detector.net.forward([self.phone_detector.net.getLayerNames()[i - 1] for i in self.phone_detector.net.getUnconnectedOutLayers()])
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    if scores[class_id] > 0.5 and self.phone_detector.classes[class_id] in ["cell phone", "laptop"]:
                        x, y, w, h = (detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
                        x -= w // 2
                        y -= h // 2
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(display_frame, self.phone_detector.classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if self.results["Spoof Detection"] == "Spoof detected" and len(self.face_detector.faces) > 0:
            x, y, w, h = self.face_detector.faces[0]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(display_frame, "SPOOF", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return display_frame

    def verify_face(self, frame):
        if frame is None or frame.size == 0:
            return "Unknown"
        self.face_detector.detect(frame)
        if len(self.face_detector.faces) != 1:
            return "Unknown"
        x, y, w, h = self.face_detector.faces[0]
        if w < 80 or h < 80:
            return "Unknown"
        face_region = frame[y:y + h, x:x + w]
        if face_region.size == 0:
            return "Unknown"
        # Optimized preprocessing
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray_face = clahe.apply(gray_face)
        gray_face = cv2.GaussianBlur(gray_face, (5, 5), 0)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        if len(eyes) >= 2:
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = sorted(eyes, key=lambda e: e[0])[:2]
            if ex1 < w and ex2 < w and ey1 < h and ey2 < h and abs(ex2 - ex1) > 10:
                eye_center1 = (int(ex1 + ew1 // 2), int(ey1 + eh1 // 2))
                eye_center2 = (int(ex2 + ew2 // 2), int(ey2 + eh2 // 2))
                angle = np.arctan2(eye_center2[1] - eye_center1[1], eye_center2[0] - eye_center1[0]) * 180 / np.pi
                center = (int(w // 2), int(h // 2))
                try:
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    gray_face = cv2.warpAffine(gray_face, M, (w, h))
                except Exception:
                    pass  # Continue without rotation
        face_processed = cv2.resize(gray_face, (100, 100), interpolation=cv2.INTER_LINEAR)
        max_confidence = 0.0
        best_result = "Unknown"
        for _ in range(20):  # Reduced retries for smoother performance
            result, confidence = self.face_recognizer.detect_with_confidence(frame)
            if result not in ["Unknown", "No Face Detected", "Multiple Faces Detected", "Face Too Small"]:
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_result = result
                if max_confidence > 0.9:
                    break
            time.sleep(0.05)  # Reduced sleep for faster processing
        if max_confidence > 0.35:  # Threshold to accept 0.37 confidence
            return best_result
        return "Unknown"

    def run(self):
        id_verified = False
        scan_started = False
        face_verification_started = False
        face_verified = False
        face_not_verified = False
        interview_finished = False
        blank_frame = np.zeros((720, 960, 3), dtype=np.uint8)
        verification_start_time = None
        self.welcome_start_time = time.time()
        while time.time() - self.welcome_start_time < 5:
            combined_frame = self.draw_ui(blank_frame, showbetter=False, initial_phase=False, show_welcome=True)
            cv2.imshow("Cheat Detection AI", combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                self.gaze_detector.close()
                self.face_detector.close()
                cv2.destroyAllWindows()
                return
        # Face verification prompt without streaming
        while not face_verification_started and not face_verified and not face_not_verified:
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            elapsed = current_time - self.last_blink_time
            if self.blink_state and elapsed >= 3.0:
                self.blink_state = False
                self.last_blink_time = current_time
            elif not self.blink_state and elapsed >= 1.0:
                self.blink_state = True
                self.last_blink_time = current_time
            # Use blank frame instead of camera feed
            combined_frame = self.draw_ui(blank_frame, showbetter=False, initial_phase=False, face_verification_phase=True)
            cv2.imshow("Cheat Detection AI", combined_frame)
            if key == ord('y'):
                face_verification_started = True
                verification_start_time = time.time()
                # Start camera only after pressing 'y'
                if not self.cap.isOpened():
                    print("Error: Camera not accessible. Reinitializing...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        print("Error: Could not reinitialize camera.")
                        self.gaze_detector.close()
                        self.face_detector.close()
                        cv2.destroyAllWindows()
                        return
            elif key == ord('q'):
                self.cap.release()
                self.gaze_detector.close()
                self.face_detector.close()
                cv2.destroyAllWindows()
                return
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Error: Camera reinitialization failed.")
                    break
                continue
            # Use original frame without heavy processing for clarity
            aligned_frame = frame.copy()
            self.face_detector.detect(aligned_frame)
            if len(self.face_detector.faces) == 1:
                x, y, w, h = self.face_detector.faces[0]
                face_region = aligned_frame[y:y + h, x:x + w]
                if face_region.size > 0:
                    # Resize with interpolation for better quality
                    face_processed = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_LINEAR)
                    face_resized = cv2.resize(face_processed, (w, h), interpolation=cv2.INTER_LINEAR)
                    aligned_frame[y:y + h, x:x + w] = face_resized
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            elapsed = current_time - self.last_blink_time
            if self.blink_state and elapsed >= 3.0:
                self.blink_state = False
                self.last_blink_time = current_time
            elif not self.blink_state and elapsed >= 1.0:
                self.blink_state = True
                self.last_blink_time = current_time
            if face_verification_started and not face_verified:
                combined_frame = self.draw_ui(aligned_frame, showbetter=False, initial_phase=False, processing_face=True)
                self.face_detector.detect(aligned_frame)
                if len(self.face_detector.faces) == 1:
                    result = self.verify_face(aligned_frame)
                    if result != "Unknown":
                        self.verified_name = result
                        self.matched_name = result
                        face_verified = True
                        face_verification_started = False
                        face_not_verified = False
                        verification_start_time = None
                    elif verification_start_time is not None and time.time() - verification_start_time >= 5.0:
                        face_not_verified = True
                        face_verification_started = False
                        verification_start_time = None
                if key == ord('t'):
                    face_not_verified = True
                    face_verification_started = False
                    verification_start_time = None
                elif key == ord('q'):
                    break
            elif face_not_verified:
                combined_frame = self.draw_ui(aligned_frame, showbetter=False, initial_phase=False, face_not_verified=True)
                if key == ord('t'):
                    face_verification_started = True
                    face_not_verified = False
                    verification_start_time = time.time()
                elif key == ord('q'):
                    break
            elif face_verified and not scan_started:
                combined_frame = self.draw_ui(aligned_frame, showbetter=False, initial_phase=False, face_verified=True, show_id_prompt=True)
                if key == ord('s'):
                    scan_started = True
                elif key == ord('q'):
                    break
            elif not id_verified:
                display_frame, id_verified = self.run_id_extraction(aligned_frame, key)
                if self.results["ID Extraction"] and "(Y/N?)" in self.results["ID Extraction"]:
                    if key == ord('y'):
                        id_verified = True
                        self.results["ID Extraction"] = self.results["ID Extraction"].replace(" (Y/N?)", "")
                        self.extracted_id = self.results["ID Extraction"]
                    elif key == ord('n'):
                        self.results["ID Extraction"] = None
                        scan_started = False
                combined_frame = self.draw_ui(display_frame, showbetter=False, initial_phase=False)
            else:
                if not self.start_time:
                    self.start_time = time.time()
                display_frame = self.run_features(aligned_frame)
                combined_frame = self.draw_ui(display_frame, showbetter=True, initial_phase=False)
            cv2.imshow("Cheat Detection AI", combined_frame)
            if key == ord('q'):
                if id_verified and not interview_finished:
                    self.save_to_sheets()
                    interview_finished = True
                break
            elif key == ord('e'):
                if id_verified and not interview_finished:
                    self.save_to_sheets()
                    interview_finished = True
                    end_frame = aligned_frame.copy()
                    cv2.putText(end_frame, "Session Ended - Data Saved", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Cheat Detection AI", end_frame)
                    cv2.waitKey(2000)
                    break
        self.cap.release()
        self.gaze_detector.close()
        self.face_detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import gspread
    system = CheatDetectionAI()
    system.run()
