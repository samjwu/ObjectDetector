import torch
import torch.nn

import mathutil


class NeuralNetwork(torch.nn.Module):
    """The convolutional neural network architecture for object detection.

    Based on Darknet, a deep learning framework used for the
    You Only Look Once (YOLO) algoritm.
    """

    def __init__(self, config_file_path: str) -> None:
        super(NeuralNetwork, self).__init__()
        self.blocks = read_configuration(config_file_path)
        self.network_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, use_cuda: bool) -> torch.Tensor:
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
                x = mathutil.predict_transform(
                    x, input_dimensions, anchors, num_classes, use_cuda
                )

                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections
