from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_name="yolo11n.pt"):
        '''
        Load the requested YOLO model.
        '''
        self.model = YOLO(model_name)

    def detect(self, frame):
        '''
        Return object labels, confidence scores and pixel boxes for one BGR frame.
        '''

        result = self.model(
            frame,
            conf=0.25,
            verbose=False
        )[0]

        detections = []

        for box in result.boxes:

            class_id = int(box.cls.item())

            detections.append({
                "label": result.names[class_id],
                "confidence": float(box.conf.item()),
                "bbox": box.xyxy[0].tolist()
            })

        return detections