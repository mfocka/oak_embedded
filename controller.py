"""
oak_pipeline.controller
~~~~~~~

Main application.

:author: Mats Fockaert
:copyright: # TODO
:license: # TODO
"""

from camera import  Camera, OAKCamera
from detection import Detection

import depthai as dai
import json

class Controller:
    """The Controller class manages the cameras and detections.
    
    It initializes the cameras based on the provided configuration,
    retrieves the detections, and manages the GUI option.
    """
    def __init__(self):
        self.hasFOD = False
        self.cam = None
        self.cap = None

    def make_cameras(self, config, cam_type="OAK Camera") -> Camera:
        """Make the cameras based on the provided configuration."""

        conf_cam_dict = {"USB Camera": "usb_camera",
                        "OAK Camera": "oak_camera", "IP Camera": "ip_camera"}
        conf_cam_type = conf_cam_dict[cam_type]
        cfg = json.load(open(config))['cameras'][conf_cam_type]
        if conf_cam_type == "oak_camera":
            self.make_camera_oak(cfg[0], conf_cam_type, id=0, cfg=cfg)
        else:
            for id, cam_config in enumerate(cfg):
                self.make_camera(cam_config, conf_cam_type, id)

    def setup_many_oak(self):
        devices = []
        for device in dai.Device.getAllAvailableDevices():
            print(
                f"[CONTROLLER]: Making device {device.getMxId()} {device.state}")
            devices.append(dai.DeviceInfo(device.getMxId()))
        return devices

    def make_camera_oak(self, config, name, id, cfg):
        if len(cfg) > 1:
            devices = self.setup_many_oak()
            for id_dev, device_info in enumerate(devices):
                cam = OAKCamera(config, name, id_dev, device_info)
                cap = cam.get_videocapture()
                # self.model.add_camera(cam, cap)
        else:
            self.cam = OAKCamera(config, name, id)
            self.cap = self.cam.get_videocapture()
            # self.model.add_camera(cam, cap)
            self.detector = Detection(self.cam, 'mobilenet-ssd')

    def get_detections(self, gui: bool):
        """Retrieve the detections and return them."""
        return self.detector.get_detections(gui)

