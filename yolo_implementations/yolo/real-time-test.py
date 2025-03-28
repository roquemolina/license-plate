from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

""" results = model.track(source='http://192.168.1.63:4747/video', show=True) """
results = model.track(source=0, show=True)

for result in results:
  boxes = result.boxes
  classes = result.names