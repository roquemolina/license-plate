"""
--------------------------------------------------------------------------------------
This file contains the first pre-idea it:
    - Read cars
    - Track cars in an object
    - Search any license plate IN THE ENTIRE FRAME,
        crop and process the image and link it
        with the car that has been added to the before.
    - Write filename, license plate and score in a csv.
    - Write in a csv to then aply the add_missing_data
      and visualize
--------------------------------------------------------------------------------------
"""

from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

import os
from datetime import datetime

# Crear directorio para guardar las placas si no existe
os.makedirs('thresholded_plates2', exist_ok=True)



results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('./../yolo/models/yolo11s.pt')
license_plate_detector = YOLO('./../yolo/models/license_plate_small_v1.pt')

# load video
cap = cv2.VideoCapture('./los-angeles.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        """ if frame_nmr > 20:
                break """
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                license_plate_crop_gray1 = cv2.resize(license_plate_crop_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                # Denoising
                license_plate_crop_gray2 = cv2.GaussianBlur(license_plate_crop_gray1, (3, 3), 0)

                license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                license_plate_crop_thresh1 = cv2.morphologyEx(license_plate_crop_thresh, cv2.MORPH_CLOSE, kernel)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                rawfilename = f"thresholded_plates2/plate_{timestamp}-raw.png"
                cv2.imwrite(rawfilename, license_plate_crop)
                filename1 = f"thresholded_plates2/plate_{timestamp}-1.png"
                cv2.imwrite(filename1, license_plate_crop_gray)
                filename2 = f"thresholded_plates2/plate_{timestamp}-2.png"
                cv2.imwrite(filename2, license_plate_crop_gray1)
                filename3 = f"thresholded_plates2/plate_{timestamp}-3.png"
                cv2.imwrite(filename3, license_plate_crop_gray2)
                filename4 = f"thresholded_plates2/plate_{timestamp}-4.png"
                cv2.imwrite(filename4, license_plate_crop_thresh)
                filename5 = f"thresholded_plates2/plate_{timestamp}-5.png"
                cv2.imwrite(filename5, license_plate_crop_thresh1)
                
                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)


                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
#write_csv(results, './los-angeles1-OCR.csv')