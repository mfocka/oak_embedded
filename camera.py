"""
oak_pipeline.camera
~~~~~~~

Main application.

:author: Mats Fockaert
:copyright: # TODO
:license: # TODO
"""

from datetime import timedelta
from pathlib import Path
import depthai as dai
import numpy as np
from typing import Tuple


class Camera:
    """The Camera class handles the camera setup and operations. This class is implemented by specific camera types and works as an abstract class."""

    def __init__(self, config: str, camera_name: str, id: int):
        self.config = config
        print('#' * 50)
        print(self.config)
        print('#' * 50)
        self.name = camera_name
        self.camera_name = f"{camera_name} {id}"
        self.id = id
        self.ranges_dict: dict[str: list[int] |
                               list[str]] = self.config["ranges"]
        self.lists_dict: dict[str: list[int] |
                              list[str]] = self.config["lists"]
        self.init_dict: dict[str: list[int] |
                             list[str]] = self.config["init_values"]
        self.parameters: list[str] = self.get_parameters()

    def get_parameters(self):
        return list(self.init_dict.keys())

    def get_videocapture(self):
        pass

    def quit(self):
        pass

    def get_frame(self):
        pass

    def isOpened(self):
        pass

    def close(self):
        pass

    def open(self):
        pass

    def setSaturation(self):
        pass

    def setGain(self):
        pass

    def setFocus(self):
        pass

    def setExposure(self):
        pass

    def setISO(self):
        pass

    def setSharpness(self):
        pass

    def setShutter(self):
        pass

    def setOptimalExposure(self):
        pass


class OAKCamera(Camera):
    """The OAKCamera class manages an OAK camera.

    This class is responsible for setting up and managing the OAK camera, 
    including setting up the camera's properties, inputs, outputs, and 
    neural network settings, and starting the pipeline. It also includes 
    methods for checking if the camera is open, quitting the camera, 
    setting initial values, and getting frames from the camera.
    """

    def __init__(self, config_location: str, camera_name: str, id: int, device_info: str = None):
        super().__init__(config_location, camera_name, id)
        self.device = None
        self.pipeline = None
        self.control = None
        self.controlQ = None
        self.device_info = device_info

        self.saturation = 0
        self.focus = 0
        self.iso = 0
        self.shutter_Us = 0
        self.sharpness = 0

        self.make_device()
        self.set_init_val()

    def isOpened(self):
        return self.device is not None

    def quit(self):
        self.control.setStopStreaming()
        self.submitControl()

    def set_init_val(self):
        if self.device is not None:
            self.control = dai.CameraControl()
            # self.control.setManualExposure(exposureTimeUs=self.init_dict['exposure'],
            #                                sensitivityIso=self.init_dict['iso'])
            self.control.setAutoFocusMode(
                dai.RawCameraControl.AutoFocusMode.OFF)
            self.control.setManualFocus(self.init_dict['focus'])
            self.control.setSaturation(self.init_dict['saturation'])
            self.control.setSharpness(self.init_dict['sharpness'])
            self.controlQ = self.device.getInputQueue('control')
            self.controlQ.send(self.control)
            print('CAMERA HAS BEEN SET UP')
        else:
            print(f"[CAMERA]: Camera could not be setup.")

    def __make_output(self):
        self.cam_out = self.pipeline.create(dai.node.XLinkOut)
        self.xout_nn = self.pipeline.create(dai.node.XLinkOut)
        self.cam_out.setStreamName("color")
        self.xout_nn.setStreamName("detections")

    def __make_properties(self, cam_rgb):
        cam_rgb.setPreviewSize(416, 416)
        cam_rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(40)
        return cam_rgb

    def __make_input(self, cam_rgb):
        controlIn = self.pipeline.create(dai.node.XLinkIn)
        controlIn.setStreamName('control')
        controlIn.out.link(cam_rgb.inputControl)
        return controlIn

    def __make_nn(self, nn, nnBlobPath):
        nn.setConfidenceThreshold(0.5)
        nn.setNumClasses(80)
        nn.setCoordinateSize(4)
        nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
        nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
        nn.setIouThreshold(0.5)
        nn.setBlobPath(nnBlobPath)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)

    def __start_pipeline(self):
        if self.device_info is None:
            try:
                self.device = dai.Device(self.pipeline)
            except:
                print('#' * 50)
                print('[ERROR]')
                self.device = None
                print('No oak device was connected.')
                print('#' * 50)
        else:
            try:
                self.device = dai.Device(self.pipeline, self.device_info)
            except:
                print('#' * 50)
                print('[ERROR]')
                self.device = None
                print(f'Oak device {self.id} was not connected.')
                print('#' * 50)

    def make_device(self):
        """Create a device based on the pipeline and device info."""

        self.pipeline = dai.Pipeline()
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        detection_nn = self.pipeline.create(dai.node.YoloDetectionNetwork)

        # Set camera output
        self.__make_output()
        # Properties
        self.__make_properties(cam_rgb)
        # Set camera input
        controlIn = self.__make_input(cam_rgb)

        nnBlobPath = str((Path(__file__).parent / Path(
            'models/yolo_v4_tiny_tf/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
        if not Path(nnBlobPath).exists():
            import sys
            raise FileNotFoundError(
                f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

        # Network specific settings
        self.__make_nn(detection_nn, nnBlobPath)
        # Link nodes
        cam_rgb.preview.link(detection_nn.input)
        cam_rgb.preview.link(self.cam_out.input)
        detection_nn.out.link(self.xout_nn.input)
        # Connect to device and start pipeline
        self.__start_pipeline()

    def get_videocapture(self):
        # Get mono camera output
        if self.device is not None:
            out = self.device.getOutputQueue(
                name="color", maxSize=4, blocking=False)
            return out
        else:
            for i in range(100):
                print('[ERROR]: NO DEVICE FOUND.')
                print(f'[ERROR]: Retry. Try # {i + 1}')
                self.make_device()
                if self.device is not None:
                    return self.get_videocapture()
            return None

    def get_frame(self):
        # Get a single frame
        out = self.get_videocapture()
        if out is not None:
            frame = out.get().getCvFrame()  # type: ignore
        else:
            print("[CAMERA]: No Frame was Found.")
            raise Exception
        return frame

    def get_video(self):
        # Obtain continuous frames for video
        out = self.get_videocapture()
        while True:
            frame = out.get().getCvFrame()  # type: ignore

            yield frame

    def getFocus(self) -> int:
        assert self.control is not None
        return self.control.getLensPosition()

    def getSaturation(self) -> int:
        return self.saturation

    def getExposure(self) -> Tuple[timedelta, int]:
        assert self.control is not None
        return (self.control.getExposureTime(), self.control.getSensitivity())

    def getSharpness(self) -> int:
        return self.sharpness

    def submitControl(self):
        assert self.control is not None
        assert self.controlQ is not None
        self.controlQ.send(self.control)

    def setFocus(self, value: int):
        assert self.control is not None
        value = np.clip(value, 0, 255)
        self.control.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.OFF)
        self.control.setManualFocus(value)
        self.submitControl()

    def setSaturation(self, value: int):
        assert self.control is not None
        self.saturation = value
        self.control.setSaturation(value)
        self.submitControl()

    def setExposureFull(self, value_ms, value_iso: int):
        assert self.control is not None
        value_iso = np.clip(value_iso, 100, 1600)
        self.control.setManualExposure(value_ms, value_iso)
        self.submitControl()

    def setOptimalExposure(self, step: int):
        if step > 5:
            print("[CAMERA]: ERROR: step is higher than amount of steps.")
        else:
            value_ms, value_iso, *_ = self.optimalExposure[str(step)]
            print(
                f"[CAMERA]: Optimal exposure for step {step}: ms: {value_ms}, iso: {value_iso}")
            self.setExposureFull(value_ms, value_iso)

    def setSharpness(self, value: int):
        assert self.control is not None
        self.sharpness = np.clip(value, 0, 4)
        self.control.setSharpness(value)
        self.submitControl()

    def setExposure(self, value: int):
        self.shutter_Us = np.clip(value, 100, 10000)
        self.setExposureFull(value, self.iso)

    def setISO(self, value: int):
        value_iso = np.clip(value, 100, 1600)
        self.iso = value_iso
        self.setExposureFull(self.shutter_Us, self.iso)

