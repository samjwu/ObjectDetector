"""Custom classes for layers in the object detector neural network"""

import torch.nn


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
