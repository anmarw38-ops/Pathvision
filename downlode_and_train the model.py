#downlode data
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="lFhKKbBEBLkOLlE45VIw")
project = rf.workspace("fsdataset").project("visdrone-r7kee")
version = project.version(1)
dataset = version.download("yolov8")


#downlode and train the model

!pip install ultralytics -q
from ultralytics import YOLO


model = YOLO("yolo11m.pt")


results = model.train(
    data="/content/VisDrone-1",
    epochs=65,
    imgsz=640,
    batch=18,
    device=0,
)
