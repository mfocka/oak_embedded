import time
from camera import OAKCamera

from pathlib import Path
import depthai as dai
import numpy as np
import cv2


class Detection:
    def __init__(self, camera: OAKCamera, model_name):
        self.camera = camera
        self.pipeline = camera.pipeline
        self.device = camera.device
        self.height = camera.get_frame().shape[0]
        self.width = camera.get_frame().shape[1]
        self.labelMap = [
            "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
            "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
            "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
            "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
            "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
            "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
            "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
            "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
            "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
            "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
            "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
            "teddy bear",     "hair drier", "toothbrush"
        ]
        self.interestedLabels = [
            "person",    "bench",  "suitcase", "backpack", "handbag",  "tie",    "bottle",
            "cup",       "fork",   "knife",    "spoon",    "bowl",     "banana", "pottedplant",
            "tvmonitor", "laptop", "mouse",    "remote",   "keyboard", "cell phone"
        ]
        self.detectionThreshold = 0.55

    def __frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def __displayFrame(self, name, bbox, label, confidence, frame):
        color = (255, 0, 0)
        cv2.putText(frame, str(
            label), (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"{int(confidence*100)}",
                    (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.rectangle(
            frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    def __get_detections(self, detections, frame, gui, windowName="rgb"):
        for detection in detections:
            try:
                label = self.labelMap[detection.label]
            except:
                label = detection.label
            if label in self.interestedLabels:
                if detection.confidence > self.detectionThreshold:
                    
                    bbox = self.__frameNorm(frame, (detection.xmin, detection.ymin,
                                            detection.xmax, detection.ymax))
                    if gui:
                        self.__displayFrame(
                            windowName, bbox, label, detection.confidence, frame)
                    print(f"{str(label)} detected at: ({bbox[0]}, {bbox[1]})")
                    print("Confidence: {:.2f}".format(
                        detection.confidence*100))
        if gui:
            cv2.imshow(windowName, frame)

    def get_detections(self, gui: bool):
        print("GETTING DETECTIONS")
        try:
            self.q_nn = self.device.getOutputQueue(
                name="detections", maxSize=31, blocking=False)
        except Exception as e:
            print("Error creating 'detections' output queue:", e)
            raise e

        detections = []
        while True:
            try:
                in_nn = self.q_nn.tryGet()

                if in_nn:
                    if self.camera.get_frame() is not None:
                        frame = self.camera.get_frame()
                        detections = in_nn.detections
                        self.__get_detections(detections, frame, gui, "rgb")
                if gui and cv2.waitKey(1) &0xff == ord('q'):
                    break

            except RuntimeError as e:
                print("RuntimeError in get_detections loop:", e)
                break
            except Exception as e:
                print("Unexpected error in get_detections loop:", e)
                break
        return detections