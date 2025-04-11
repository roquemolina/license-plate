"""
--------------------------------------------------------------------------------------
This file contains the first v1 it:
    - Take a video as an input and read objects
    - Filter cars and detection confidence is > 60%
    - Search any license plate INSIDE COOPRED CAR FRAME and 
      detection confidence is > 60%
    - Process cropped license plate
    - Read license plate text and
      detection confidence is > 60%
    - Send to API license plate
--------------------------------------------------------------------------------------
"""

from ultralytics import YOLO
import cv2
import re
from datetime import datetime
import easyocr
import signal

# Graceful exit setup
exit_flag = False

def signal_handler(sig, frame):
    global exit_flag
    print("\nCtrl+C pressed. Exiting gracefully...")
    exit_flag = True

signal.signal(signal.SIGINT, signal_handler)

# Crear directorio para guardar las placas si no existe
#os.makedirs('reader_v1_imgs2', exist_ok=True)

import requests
from datetime import datetime

# load models
coco_model = YOLO('./../yolo/models/yolo11s.pt')
license_plate_detector = YOLO('./../yolo/models/license_plate_small_v1.pt')

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Define allowed characters (uppercase letters and numbers)
ALLOWED_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

# load video / 0 => camera
cap = cv2.VideoCapture(0)

vehicles = [2, 3, 5, 7]



while not exit_flag:

    ret, frame = cap.read()
    if not ret:
        print("Error reading frame - check camera connection")
        break

    # Process every nth frame
    frame_nmr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_nmr += 1
    if frame_nmr % 3 != 0:  # Process 1/3 of frames to reduce load
        continue

    # detect vehicles
    vehicle_detections = coco_model(frame)[0]
    detections_ = []

    for detection in vehicle_detections.boxes.data.tolist():
        x1, y1, x2, y2, detection_score, class_id = detection
        # Filter by confidence score (60% threshold) and vehicle
        if int(class_id) in vehicles and detection_score > 0.6:
          detections_.append([x1, y1, x2, y2, detection_score])
        if int(class_id) == 58:
          api_url = "http://localhost:3000/save"
          payload = {
            "license_plate": "Potted!",
            "score": "0.2"
            }
          try:
            response = requests.post(api_url, json=payload, timeout=2)
            if response.status_code != 200:
                print(f"API Error: {response.json().get('message', 'Unknown error')}")
            print(f"License plate succefully sent")
          except Exception as e:
            print(f"API Connection Failed: {str(e)}")

    for vehicle in detections_:
        xcar1, ycar1, xcar2, ycar2, car_score = vehicle

        # Crop the vehicle region from the frame
        car_crop = frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2)]

        # Detect license plates WITHIN THE CROPPED CAR REGION
        license_plates = license_plate_detector(car_crop)[0]

        for plate in license_plates.boxes.data.tolist():
          x1, y1, x2, y2, plate_score, _ = plate

          # Skip plates with confidence below 60%
          if plate_score < 0.6:  
            continue

          # Crop the license plate from the car region
          plate_crop = car_crop[int(y1):int(y2), int(x1):int(x2)]

          # Focus on the text region (exclude top 40% for state logo)
          # Calculate crop adjustments
          plate_height = y2 - y1
          plate_width = x2 - x1
          
          # Remove 30% from top, 5% from left/right, 10% from bottom
          y_start = int(y1 + 0.20 * plate_height)
          y_end = int(y2 - 0.10 * plate_height)
          x_start = int(x1 + 0.04 * plate_width)
          x_end = int(x2 - 0.04 * plate_width)
          
          # Apply adjusted crop (with boundary checks)
          text_region = plate_crop[
              max(0, y_start - int(y1)):min(plate_crop.shape[0], y_end - int(y1)),
              max(0, x_start - int(x1)):min(plate_crop.shape[1], x_end - int(x1))
          ]

            # process license plate
          text_region_gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
          text_region_gray1 = cv2.resize(text_region_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
          # Denoising
          text_region_gray2 = cv2.GaussianBlur(text_region_gray1, (3, 3), 0)

          text_region_thresh = cv2.adaptiveThreshold(text_region_gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
          
          kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
          text_region_thresh1 = cv2.morphologyEx(text_region_thresh, cv2.MORPH_CLOSE, kernel)
          
          plate_texts = reader.readtext(text_region_thresh1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

          for plate_text in plate_texts:
            
            bbox, text, text_score = plate_text
            # Skip plates with confidence below 60%
            if text_score < 0.6:  
              continue
              
            # Clean the text: uppercase, remove spaces, and filter special characters
            cleaned_text = text.upper().replace(' ', '')
              # Remove any remaining non-alphanumeric characters using regex
            cleaned_text = re.sub(f'[^{"".join(ALLOWED_CHARS)}]', '', cleaned_text)
            # Only send if we have text left after cleaning and 5-9 char length
            if cleaned_text and 5 <= len(cleaned_text) <= 9:
                api_url = "http://localhost:3000/save"
                payload = {
                    "license_plate": cleaned_text,
                    "score": round(text_score, 4)
                  }
                try:
                    response = requests.post(api_url, json=payload, timeout=2)
                    if response.status_code != 200:
                        print(f"API Error: {response.json().get('message', 'Unknown error')}")
                    print(f"License plate succefully sent")
                except Exception as e:
                  print(f"API Connection Failed: {str(e)}")
# Cleanup
cap.release()
try:
    cv2.destroyAllWindows()  # Safe to call even if no windows exist
except:
    pass
print("Resources released")