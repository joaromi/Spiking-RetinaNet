# Spiking-RetinaNet

This repository corresponds to the research documented in: [RetinaNet Object Detector based on Analog-to-Spiking Neural Network Conversion](https://arxiv.org/abs/2106.05624).

The code stems from the library [SNN_Toolbox](https://github.com/NeuromorphicProcessorProject/snn_toolbox), modifying it to address object detection problems.
As a trade-off many side features of the original library have been removed. The conversion algorithm only accepts a Keras input model, and sticks to a _'temporal_mean_rate'_ encoding for the spiking activations and a _'constant_input_currents'_ encoding for the input images.

Configuration used:
*Python       3.8.6
*tensorflow   2.3.1
*Keras        2.4.3
