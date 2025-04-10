""" import easyocr
import csv
import os
import glob

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Get list of PNG files in the thresholded_plates directory
image_files = glob.glob(os.path.join('thresholded_plates', '*.png'))

# Create and write to CSV file
with open('plate_results.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header
    csv_writer.writerow(['Filename', 'Text', 'Confidence'])
    
    # Process each image file
    for img_path in image_files:
        # Get just the filename without path
        filename = os.path.basename(img_path)
        
        # Perform OCR
        detections = reader.readtext(img_path)
        
        # Write each detection to CSV
        for detection in detections:
            bbox, text, score = detection
            cleaned_text = text.upper().replace(' ', '')
            csv_writer.writerow([filename, cleaned_text, round(score, 4)])

print("Processing complete. Results saved to plate_results.csv") """

import easyocr
import csv
import os
import glob
import re

# Initialize the OCR reader with English
reader = easyocr.Reader(['en'], gpu=True)

# Get list of PNG files
image_files = glob.glob(os.path.join('thresholded_plates2', '*.png'))

# Define allowed characters (uppercase letters and numbers)
ALLOWED_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

with open('plate_results-allowed-chars1.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Filename', 'Text', 'Confidence'])
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # OCR with focus on allowed characters
        detections = reader.readtext(img_path, 
                                  allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        for detection in detections:
            bbox, text, score = detection
            
            # Clean the text: uppercase, remove spaces, and filter special characters
            cleaned_text = text.upper().replace(' ', '')
            # Remove any remaining non-alphanumeric characters using regex
            cleaned_text = re.sub(f'[^{"".join(ALLOWED_CHARS)}]', '', cleaned_text)
            
            if cleaned_text:  # Only write if we have text left after cleaning
                csv_writer.writerow([filename, cleaned_text, round(score, 4)])

print("Processing complete. Results saved to plate_results.csv")