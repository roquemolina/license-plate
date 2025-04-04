from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

# Train the model
# Added if __name__ == "__main__": as documentation says it could cause problems in windows
if __name__ == "__main__":
  results = model.train(
    data="config.yaml",           # Switch to FULL COCO (not coco8)
    epochs=100,                 # Longer training for better convergence
    imgsz=640,                  
    batch=24,                   # Max batch size without OOM errors
    device=0,
    pretrained=True,
    optimizer="AdamW",          # Explicitly set better optimizer
    lr0=0.001,                  # Learning rate (lower for fine-tuning)
    cos_lr=True,                # Cosine learning rate scheduler
    augment=True,               # Enable mosaics
    mixup=0.2,                  # Advanced augmentation
    flipud=0.5,                 # Simulate upside-down objects
    degrees=45,                 # Rotation augmentation
    name="license_plate_small_v1"
  )