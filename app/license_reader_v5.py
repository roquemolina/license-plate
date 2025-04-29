import cv2
import os
import re
import signal
import threading
import queue
import requests
import torch
import paddle
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --------- Configuration ---------
INPUT_SOURCE = './chacabuco-phone.mp4'
FRAME_SKIP = 3  # Process every 3rd frame
CONFIDENCE_THRESH = 0.6
ALLOWED_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
OUTPUT_DIR = datetime.now().strftime("./detections_output-chacabuco_v5")
API_URL = "http://localhost:3000/save"

# --------- Environment Setup ---------
os.makedirs(OUTPUT_DIR, exist_ok=True)
exit_flag = False

# Device Detection
YOLO_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OCR_USE_GPU = paddle.is_compiled_with_cuda()

print(f"YOLO device: {YOLO_DEVICE}")
print(f"PaddleOCR using GPU: {OCR_USE_GPU}")

# --------- Model Initialization ---------
print("Loading YOLO models...")
coco_model = YOLO('./../yolo/models/yolo11s.pt').to(YOLO_DEVICE)
license_plate_model = YOLO('./../yolo/models/license_plate_small_v1.pt').to(YOLO_DEVICE)

print("Loading PaddleOCR...")
if OCR_USE_GPU:
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
else:
    paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, det_db_box_thresh=0.5)

# Graceful exit on Ctrl+C
def signal_handler(sig, frame):
    global exit_flag
    print("\nCtrl+C detected. Exiting...")
    exit_flag = True

signal.signal(signal.SIGINT, signal_handler)

# --------- Helper Functions ---------
def clean_plate_text(text):
    text = text.upper().replace(' ', '')
    text = re.sub(f'[^{"".join(ALLOWED_CHARS)}]', '', text)
    return text

def save_images(cleaned_text, frame, car_crop, plate_crop):
    try:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{cleaned_text}.jpg"), frame)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"car_{cleaned_text}.jpg"), car_crop)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"plate_{cleaned_text}.jpg"), plate_crop)
    except Exception as e:
        print(f"Error saving images: {e}")

def send_plate_to_api(cleaned_text, text_score):
    payload = {"license_plate": cleaned_text, "score": round(text_score, 4)}
    try:
        response = requests.post(API_URL, json=payload, timeout=2)
        if response.status_code == 200:
            print(f"[API] License plate {cleaned_text} sent successfully!")
        else:
            print(f"[API] Error: {response.text}")
    except Exception as e:
        print(f"[API] Connection failed: {e}")

# --------- Worker Thread for OCR and API ---------
def ocr_worker(plate_queue):
    while not exit_flag or not plate_queue.empty():
        try:
            plate_crop, frame, car_crop = plate_queue.get(timeout=1)
        except queue.Empty:
            continue

        h, w, _ = plate_crop.shape
        refined_crop = plate_crop[int(0.2*h):int(0.9*h), int(0.05*w):int(0.95*w)]

        # Preprocessing for OCR (following v4)
        gray = cv2.cvtColor(refined_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        ocr_results = paddle_ocr.ocr(closed, cls=True)

        if not ocr_results:
            continue

        for group in ocr_results:
            if group is None:
                continue
            for box in group:
                text, text_score = box[1]
                if text_score < CONFIDENCE_THRESH:
                    continue
                cleaned_text = clean_plate_text(text)
                if 4 <= len(cleaned_text) <= 9:
                    print(f"[Detected] Plate: {cleaned_text} (Confidence: {round(text_score,2)})")
                    save_images(cleaned_text, frame, car_crop, plate_crop)
                    send_plate_to_api(cleaned_text, text_score)

# --------- Main Video Loop ---------

plate_queue = queue.Queue()
threading.Thread(target=ocr_worker, args=(plate_queue,), daemon=True).start()

cap = cv2.VideoCapture(INPUT_SOURCE)
if not cap.isOpened():
    print("Error: Could not open input.")
    exit()

vehicles = [2, 3, 5, 7]  # Vehicle class IDs

while not exit_flag:
    ret, frame = cap.read()
    if not ret:
        print("End of video or camera feed lost.")
        break

    frame_nmr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_nmr % FRAME_SKIP != 0:
        continue

    # Vehicle Detection
    vehicle_detections = coco_model(frame)[0]
    valid_vehicles = [d for d in vehicle_detections.boxes.data.tolist() if int(d[5]) in vehicles and d[4] > CONFIDENCE_THRESH]

    for x1, y1, x2, y2, car_score, _ in valid_vehicles:
        car_crop = frame[int(y1):int(y2), int(x1):int(x2)]

        # License Plate Detection within car crop
        plates = license_plate_model(car_crop)[0]

        for px1, py1, px2, py2, plate_score, _ in plates.boxes.data.tolist():
            if plate_score < CONFIDENCE_THRESH:
                continue

            plate_crop = car_crop[int(py1):int(py2), int(px1):int(px2)]

            # Queue plate for OCR processing
            plate_queue.put((plate_crop, frame.copy(), car_crop.copy()))

cap.release()
try:
    cv2.destroyAllWindows()
except:
    pass

print("\nAll resources released. Program terminated.")