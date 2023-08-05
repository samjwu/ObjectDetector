"""The convolutional neural network architecture for object detection."""

import numpy
import torch
import torch.nn

import fileutil
import mathutil


class NeuralNetwork(torch.nn.Module):
    """Convolutional neural network.

    Based on Darknet, a deep learning framework used for the
    You Only Look Once (YOLO) algoritm.
    """

    def __init__(self, config_file_path: str) -> None:
        super(NeuralNetwork, self).__init__()
        self.blocks = fileutil.read_configuration(config_file_path)
        self.network_info, self.module_list = fileutil.create_modules(self.blocks)

    def forward(self, x: torch.Tensor, use_cuda: bool) -> torch.Tensor:
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
                x = self.module_list[idx](x)

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
                input_dimensions = int(self.network_info["height"])
                anchors = self.module_list[idx][0].anchors
                num_classes = int(module["classes"])
                x = x.data
                x = mathutil.predict_transform(
                    x, input_dimensions, anchors, num_classes, use_cuda
                )

                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[idx] = x

        return detections

    def load_weights(self, weight_file: str) -> None:
        """Load in the weights used for the convolutional neural network.

        The input is multiplied by the weights
        in the hidden layers when determining the output.
        """
        weight_file_object = open(weight_file, "rb")

        # get header information from first 5 values in weight file
        # indices 0-3: major.minor.patch version, respectively
        # indices 4/5: images seen by the network during training
        header = numpy.fromfile(weight_file_object, dtype=numpy.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen_images = self.header[3]

        weights = numpy.fromfile(weight_file_object, dtype=numpy.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # only load weights for convolutional module
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # the number of weights in the Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # load the weights
                    bn_biases = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr : ptr + num_bn_biases]
                    )
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr : ptr + num_bn_biases]
                    )
                    ptr += num_bn_biases

                    # reshape loaded weights into dimensions of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy the data to the model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # the number of weights
                    num_biases = conv.bias.numel()

                    # load the weights
                    conv_biases = torch.from_numpy(weights[ptr : ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape loaded weights into dimensions of model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # copy the data to the model
                    conv.bias.data.copy_(conv_biases)

                # load the weights for convolutional layers
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr : ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
