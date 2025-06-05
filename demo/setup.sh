#!/bin/bash
# -*- coding: utf-8 -*-
# File: setup.sh
# This script sets up the environment for the demo.

# Install pytorch - https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

# Install other libraries
pip install matplotlib scipy transformers ultralytics albumentations segmentation_models_pytorch optimum[onnxruntime]