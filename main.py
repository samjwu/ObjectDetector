import torch
import torch.nn


class NeuralNetwork(torch.nn.Module):
    """The convolutional neural network architecture for object detection.

    Based on Darknet, a deep learning framework used for the
    You Only Look Once (YOLO) algoritm.
    """

    def __init__(self, config_file_path: str) -> None:
        super(NeuralNetwork, self).__init__()
        self.blocks = read_configuration(config_file_path)
        self.network_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA: bool) -> torch.Tensor:
        """Performs computation for each forward pass.

        Iterate over modules, concatenating feature maps for each layer.
        """
        modules = self.blocks[1:]
        # cache for route and shortcut layers
        outputs = {}  # outputs[layer index] = feature map
        write = False

        for idx, module in enumerate(modules):
            module_type = module["type"]

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(layer) for layer in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - idx

                if len(layers) == 1:
                    x = outputs[idx + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - idx

                    map1 = outputs[idx + layers[0]]
                    map2 = outputs[idx + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_layer = int(module["from"])
                x = outputs[idx - 1] + outputs[idx + from_layer]

            elif module_type == "yolo":
                input_dimensions = int(self.net_info["height"])
                anchors = self.module_list[i][0].anchors
                num_classes = int(module["classes"])
                x = x.data
                x = predict_transform(x, input_dimensions, anchors, num_classes, CUDA)

                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections


class EmptyLayer(torch.nn.Module):
    """Placeholder layer.

    Used by route layers and skip connections.
    Executed later by the forward method defined here,
    after calling the forward method from torch.nn.Module
    to concatenate feature maps.
    """

    def __init__(self) -> None:
        super(EmptyLayer, self).__init__()


class DetectionLayer(torch.nn.Module):
    """Detection layer.

    Used by YOLO layer.
    Holds anchors for detecting bounding boxes.
    """

    def __init__(self, anchors: list[int]) -> None:
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


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
                    block.clear()

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
    filters = 3
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
                batch_normalize_layer = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), batch_normalize_layer)

            if activation == "leaky":
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
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

            route = EmptyLayer()

            module.add_module("route_{0}".format(index), route)

            if route_end < 0:
                filters = (
                    output_filters[index + route_start]
                    + output_filters[index + route_end]
                )
            else:
                filters = output_filters[index + route_start]

        elif block["type"] == "shortcut":  # skip connection
            shortcut = EmptyLayer()

            module.add_module("shortcut_{}".format(index), shortcut)

        elif block["type"] == "yolo":  # detection layer
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)

            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (network_info, module_list)


config_file_path = "yolo.cfg"
blocks = read_configuration(config_file_path)
network_info, module_list = create_modules(blocks)
