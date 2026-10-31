from ultralytics import YOLO
import config


class Detector:
     def __init__(self):
        self.model = YOLO(config.MODEL_PATH)

     def track(self, frame):
        results = self.model.track(frame, persist=True, verbose=False, classes=[0])  #Sadece insanları tespit et.
        return results[0]


        
     

     