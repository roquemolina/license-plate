from ultralytics import YOLO

""" yolo detect train data=config.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0 """

# Load a model
model = YOLO("yolov8n.yaml")

# Train the model
train_results = model.train(
    data="config.yaml",  # path to dataset YAML
    epochs=2,  # number of training epochs
    imgsz=640,  # training image size
    #device="gpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    #device=0,
).to('cuda:0')