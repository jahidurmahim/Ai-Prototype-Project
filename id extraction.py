# Extracts and validates 16-digit ID numbers from images using OCR, with scan retries, threading, and user confirmation handling.

import cv2
import pytesseract
import re
import threading
import time
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class IDExtractor:
    # Initializes the ID extractor and sets default values for tracking scan state
    def __init__(self):
        self.id_result = None
        self.result_frame = None
        self.processing = False
        self.last_scan_time = 0
        self.scan_cooldown = 2
        self.scan_attempts = 0
        self.max_attempts = 5
        self.id_verified = False
        self.waiting_for_confirmation = False
        self.rejection_reason = ""

    # Checks if the given ID number is valid (must be 16 digits and start with '022222')
    def validate_id_number(self, id_number):
        return id_number and len(id_number) == 16 and id_number.startswith('022222')

    # Extracts a valid ID number from an image using OCR
    def extract_id_number(self, image):
        img = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=config)
        all_digits = ''.join(filter(str.isdigit, text))
        pattern = r'022222\d{10}'
        match = re.search(pattern, all_digits)
        return match.group(0) if match else all_digits if len(all_digits) == 16 else ""

    # Runs the ID extraction in a background thread and updates result
    def process_id_in_thread(self, roi, full_frame):
        self.processing = True
        id_number = self.extract_id_number(roi)
        result_frame = roi.copy()

        if self.validate_id_number(id_number):
            self.id_result = id_number
            self.waiting_for_confirmation = True
        else:
            id_number = self.extract_id_number(full_frame)
            if self.validate_id_number(id_number):
                self.id_result = id_number
                self.waiting_for_confirmation = True
            else:
                self.scan_attempts += 1
                self.id_result = None
                self.rejection_reason = "ID not detected"

        self.result_frame = result_frame
        self.processing = False

    # Handles user input, manages scan cooldowns, and returns ID extraction results
    def process_frame(self, frame, force_scan=False, key_pressed=None):
        display_frame = frame.copy()

        if self.waiting_for_confirmation:
            if key_pressed in [ord('y'), ord('Y')]:
                self.id_verified = True
                self.waiting_for_confirmation = False
            elif key_pressed in [ord('n'), ord('N')]:
                if self.scan_attempts > 0:
                    self.scan_attempts -= 1
                self.waiting_for_confirmation = False
                self.id_result = None
                self.rejection_reason = "User rejected"

        current_time = time.time()
        if not self.id_verified and not self.processing and force_scan and (
                current_time - self.last_scan_time) > self.scan_cooldown:
            if self.scan_attempts < self.max_attempts:
                h, w = frame.shape[:2]
                roi = frame[int(h / 4):int(3 * h / 4), int(w / 4):int(3 * w / 4)].copy()
                threading.Thread(target=self.process_id_in_thread, args=(roi, frame), daemon=True).start()
                self.last_scan_time = current_time

        return display_frame, self.id_result, self.id_verified
