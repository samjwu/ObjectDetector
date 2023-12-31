"""Test code for checking neural network functions."""

import torch

import fileutil
import neuralnetwork


config_file_path = "data/yolo.cfg"
image_file_path = "data/test.png"
weights_file_path = "data/yolov3.weights"
blocks = fileutil.read_configuration(config_file_path)
network_info, module_list = fileutil.create_modules(blocks)

model = neuralnetwork.NeuralNetwork(config_file_path)
model.load_weights(weights_file_path)
model_input = fileutil.process_input_image(image_file_path)
prediction = model(model_input, False)
print(prediction)
