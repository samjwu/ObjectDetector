"""Utility functions for reading files."""

import cv2
import numpy
import torch.autograd
import torch.nn

import layers
import mathutil


def read_configuration(config_file_path: str) -> list[dict[str, str]]:
    """Reads a file and return configurations for each layer.

    Returns the information for each layer as a block.

    Args:
        config_file_path (str): Path to the configuration file.

    Returns:
        blocks (list[dict[str, str]]): A list of blocks
            representing the information for each layer.
    """
    blocks = []

    with open(config_file_path, "r") as file:
        lines = file.read().split("\n")
        configs = []

        for line in lines:
            # skip comments and empty lines
            if len(line) == 0 or line[0] == "#":
                continue
            else:
                configs.append(line.lstrip().rstrip())

        block = {}

        for config in configs:
            # new block
            if config[0] == "[":
                # add previous block to list of blocks
                if block:
                    blocks.append(block)
                    block = {}

                block["type"] = config[1:-1].rstrip()
            else:
                key, value = config.split("=")
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)

    return blocks


def create_modules(
    blocks: list[dict[str, str]]
) -> tuple[dict[str, str], torch.nn.ModuleList]:
    """Create modules from block information.

    For convolutional layers, add the batch normalization and
    leaky Rectified Linear Unit (ReLU) layers, if present, to the same module.

    Args:
        blocks (list[dict[str, str]]): List of configurations for the layers.

    Returns:
        network_info (tuple[dict[str, str]): Contains the
            inputs to the network and training parameters.
        module_list (torch.nn.ModuleList): List of layers to be used
            in the neural network.
    """
    network_info = blocks[0]
    module_list = torch.nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, block in enumerate(blocks[1:]):
        # create a new module for each block
        module = torch.nn.Sequential()

        if block["type"] == "convolutional":  # convolutional layer
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            padding = int(block["pad"])
            activation = block["activation"]

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            convolutional_layer = torch.nn.Conv2d(
                in_channels=prev_filters,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=bias,
            )
            module.add_module("conv_{0}".format(index), convolutional_layer)

            if batch_normalize:
                batch_normalize_layer = torch.nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), batch_normalize_layer)

            if activation == "leaky":
                activation_layer = torch.nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activation_layer)

        elif block["type"] == "upsample":  # upsample layer
            stride = int(block["stride"])

            upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

            module.add_module("upsample_{}".format(index), upsample)

        elif block["type"] == "route":  # route layer
            block["layers"] = block["layers"].split(",")

            route_start = int(block["layers"][0])

            try:
                route_end = int(block["layers"][1])
            except:
                route_end = 0

            # if value is positive, subtract current index
            # (to account for negative values)
            if route_start > 0:
                route_start = route_start - index
            if route_end > 0:
                route_end = route_end - index

            route = layers.EmptyLayer()

            module.add_module("route_{0}".format(index), route)

            if route_end < 0:
                filters = (
                    output_filters[index + route_start]
                    + output_filters[index + route_end]
                )
            else:
                filters = output_filters[index + route_start]

        elif block["type"] == "shortcut":  # skip connection
            shortcut = layers.EmptyLayer()

            module.add_module("shortcut_{}".format(index), shortcut)

        elif block["type"] == "yolo":  # detection layer
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = layers.DetectionLayer(anchors)

            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (network_info, module_list)


def process_input_image(file: str):
    """Read and process an image file."""
    img = cv2.imread(file)

    img = cv2.resize(img, (416, 416))

    # transpose BGR to RGB  (height, width, color -> color, height, width)
    processed_img = img[:, :, ::-1].transpose((2, 0, 1))

    # add a dimension at index 0 for batches
    processed_img = processed_img[numpy.newaxis, :, :, :] / 255.0

    processed_img = torch.from_numpy(processed_img).float()

    # tensor wrapper for computing gradients
    processed_img = torch.autograd.Variable(processed_img)

    return processed_img


def determine_output(
    prediction,
    confidence: float,
    num_classes: int,
    non_maximum_suppression_confidence: int = 0.4,
) -> torch.Tensor:
    """Calculate and return the output."""
    # set prediction attributes to zero
    # for bounding boxes below the confidence threshold
    confidence_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * confidence_mask

    # get bounding box corner coordinates from bounding box attributes
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False

    for index in range(batch_size):
        image_prediction = prediction[index]

        # obtain index of class with highest confidence score
        max_confidence, max_confidence_score = torch.max(
            image_prediction[:, 5 : 5 + num_classes], 1
        )
        max_confidence = max_confidence.float().unsqueeze(1)
        max_confidence_score = max_confidence_score.float().unsqueeze(1)
        sequence = (image_prediction[:, :5], max_confidence, max_confidence_score)
        image_prediction = torch.cat(sequence, 1)

        # remove bounding boxes with confidence below threshold
        non_zero_index = torch.nonzero(image_prediction[:, 4])
        try:
            image_prediction_ = image_prediction[non_zero_index.squeeze(), :].view(
                -1, 7
            )
        except:
            continue

        if image_prediction_.shape[0] == 0:
            continue

        # get unique classes detected in the image
        img_classes = mathutil.unique(
            image_prediction_[:, -1]
        )  # index -1 to hold the class index

        # apply non maximum suppression to each image class
        for img_class in img_classes:
            # get image detections for a class
            class_mask = image_prediction_ * (
                image_prediction_[:, -1] == img_class
            ).float().unsqueeze(1)
            class_mask_index = torch.nonzero(class_mask[:, -2]).squeeze()
            image_prediction_class = image_prediction_[class_mask_index].view(-1, 7)

            # sort detections by confidence score descending
            confidence_sort_index = torch.sort(
                image_prediction_class[:, 4], descending=True
            )[1]

            # get the number of detections
            image_prediction_class = image_prediction_class[confidence_sort_index]
            num_detections = image_prediction_class.size(0)

            for i in range(num_detections):
                # calculate the intersection over union
                # for all boxes after the current one
                try:
                    intersection_over_union = (
                        calculate_bounding_box_intersection_over_union(
                            image_prediction_class[i].unsqueeze(0),
                            image_prediction_class[i + 1 :],
                        )
                    )
                except ValueError:
                    break

                except IndexError:
                    break

                # set all detections with intersection over union
                # higher than the non maximum suppression confidence threshold
                # to zero
                intersection_over_union_mask = (
                    (intersection_over_union < non_maximum_suppression_confidence)
                    .float()
                    .unsqueeze(1)
                )
                image_prediction_class[i + 1 :] *= intersection_over_union_mask

                # remove non-zero entries
                non_zero_index = torch.nonzero(image_prediction_class[:, 4]).squeeze()
                image_prediction_class = image_prediction_class[non_zero_index].view(
                    -1, 7
                )

            # repeat the batch id for each detection of the img_class in the image
            batch_index = image_prediction_class.new(
                image_prediction_class.size(0), 1
            ).fill_(index)
            sequence = batch_index, image_prediction_class

            if not write:
                output = torch.cat(sequence, 1)
                write = True
            else:
                out = torch.cat(sequence, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0
