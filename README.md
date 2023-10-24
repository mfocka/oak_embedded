# oak_embedded

## Introduction

Example application for creating an object detection pipeline using OAK-1 and an imx8plus board.

## Error

When we call for the detection (in "detection.py", line 83: `in_nn = self.q_nn.tryGet()`), we get the following error (only on the board, not on a pc): `RuntimeError: Communication exception - possible device error/misconfiguration. Original message 'Couldn't read data from stream: 'input' (X_LINK_ERROR)`.

This error, as stated, does not happen on other devices. Tried running this on Windows and on Ubuntu, both worked fine.

## Installed packages

- depthai
- opencv-python-headless

## Hardware

[imx8plus](https://embedded.avnet.com/product/msc-sm2s-imx8plus/) running Yocto Kirkstone with kernel 5.15.52.