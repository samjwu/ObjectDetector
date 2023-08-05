import argparse

import torch

import fileutil
import neuralnetwork


def arg_parse():
    """Process flags passed to the object detector module."""

    parser = argparse.ArgumentParser(description="Object Detector Module")

    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        help="Image or directory containing input images used for detection.",
        default="data",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Image or directory to to which detections are output.",
        default="output",
        type=str,
    )
    parser.add_argument("-b", "--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument(
        "-f",
        "--confidence",
        dest="confidence",
        help="Object confidence to filter predictions. \
            Class confidences represent the probabilities of a detected object \
            belonging to a particular class. \
            Higher values mean a higher probability of an object detected.",
        default=0.5,
    )
    parser.add_argument(
        "-n",
        "--nms",
        dest="nms",
        help="Non-maximum suppression threshhold. \
            Used to select predictions with confidence above the threshold \
            and suppress predictions below the threshold.",
        default=0.4,
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="configfile",
        help="Configuration file. \
            Contains information about layers \
            used for the object detector's convolutional neural network.",
        default="config/yolo.cfg",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weights",
        dest="weightsfile",
        help="Weights file. \
            Weight is a parameter in a neural network \
            that transforms input data in the network's hidden layers. \
            When an input enters a node/neuron, it is multiplied by a weight value. \
            The output is observed or passed to the next layer in the neural network.",
        default="data/yolov3.weights",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--res",
        dest="res",
        help="Input resolution of the network. \
            Higher values increase output accuracy, lower values increase output speed.",
        default="416",
        type=str,
    )

    return parser.parse_args()


config_file_path = "config/yolo.cfg"
image_file_path = "data/test.png"
weights_file_path = "data/yolov3.weights"
blocks = fileutil.read_configuration(config_file_path)
network_info, module_list = fileutil.create_modules(blocks)

model = neuralnetwork.NeuralNetwork(config_file_path)
model.load_weights(weights_file_path)
model_input = fileutil.process_input_image(image_file_path)
# if torch.cuda.is_available():
#     prediction = model(model_input.cuda(), True)
# else:
#     prediction = model(model_input, False)
prediction = model(model_input, False)
print(prediction)
