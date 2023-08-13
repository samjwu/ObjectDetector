"""Main code for object detector module."""

import argparse
import os
import os.path
import time

import cv2
import torch

import fileutil
import mathutil
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
        default="data/yolo.cfg",
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

# load network
load_start_time = time.time()
model = neuralnetwork.NeuralNetwork(args.configfile)
model.load_weights(args.weights)
load_end_time = time.time()
print(f"Time to load neural network: {load_end_time - load_start_time} seconds")

# get network information
model.network_info["height"] = args.res
input_dimensions = int(model.network_info["height"])
print(f"Height: {input_dimensions}")

if using_cuda:
    model.cuda()

# set model in evaluation mode
model.eval()

# read paths to images
read_start_time = time.time()

try:
    # check for folder of images
    image_list = [
        os.path.join(os.path.realpath("."), images, img) for img in os.listdir(images)
    ]
except NotADirectoryError:
    # check for single image
    image_list = []
    image_list.append(os.path.join(osp.realpath("."), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

read_end_time = time.time()
print(f"Time to read input images: {read_end_time - read_start_time} seconds")

# create output directory
if not os.path.exists(args.output):
    os.makedirs(args.output)

# load images
load_batch_start_time = time.time()
loaded_images = [cv2.imread(x) for x in image_list]

# prepare images in loaded images
image_batches = list(
    map(
        mathutil.prepare_image,
        loaded_images,
        [image_input_dimensions for x in range(len(image_list))],
    )
)

# get list with dimensions of original images
image_dimension_list = [(x.shape[1], x.shape[0]) for x in loaded_images]
image_dimension_list = torch.FloatTensor(image_dimension_list).repeat(1, 2)

if use_cuda:
    image_dimension_list = image_dimension_list.cuda()

# check if there is an extra batch due to remainder
has_remainder = 0
if len(image_dimension_list) % batch_size:
    has_remainder = 1

# create image batches
if batch_size != 1:
    num_batches = len(image_list) // batch_size + has_remainder
    image_batches = [
        torch.cat(
            (
                image_batches[
                    i * batch_size : min((i + 1) * batch_size, len(image_batches))
                ]
            )
        )
        for i in range(num_batches)
    ]
