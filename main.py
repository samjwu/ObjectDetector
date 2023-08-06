import argparse

import torch

import fileutil
import neuralnetwork


def parse_arguments():
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
    parser.add_argument(
        "-b",
        "--batchsize",
        dest="batchsize",
        help="Batch size. \
            Number of training samples in one forward or backward pass. \
            Batch size defines the number of samples \
            that will be propagated through the network each iteration.",
        default=1,
    )
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
        "-m",
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
        dest="weights",
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
    parser.add_argument(
        "-n",
        "--names",
        dest="names",
        help="File containing names of objects in the dataset.",
        default="data/coco.names",
        type=str,
    )

    return parser.parse_args()


# check if GPU supports CUDA
using_cuda = torch.cuda.is_available()

# get input image(s)
args = parse_arguments()
images = args.input

# convolutional neural network configuration
batch_size = int(args.batchsize)
object_confidence = float(args.confidence)
nms_threshold = float(args.nms)

# COCO dataset configuration
num_classes = 80
names = args.names
classes = fileutil.load_classes(names)

print("Loading neural network...")
model = neuralnetwork.NeuralNetwork(args.configfile)
model.load_weights(args.weights)
print("Neural network done loading.")

model.network_info["height"] = args.res
input_dimensions = int(model.network_info["height"])
print(f"Height: {input_dimensions}")

if using_cuda:
    model.cuda()

# set model in evaluation mode
model.eval()

# config_file_path = "config/yolo.cfg"
# image_file_path = "data/test.png"
# weights_file_path = "data/yolov3.weights"
# blocks = fileutil.read_configuration(config_file_path)
# network_info, module_list = fileutil.create_modules(blocks)

# model = neuralnetwork.NeuralNetwork(config_file_path)
# model.load_weights(weights_file_path)
# model_input = fileutil.process_input_image(image_file_path)
# # if torch.cuda.is_available():
# #     prediction = model(model_input.cuda(), True)
# # else:
# #     prediction = model(model_input, False)
# prediction = model(model_input, False)
# print(prediction)
