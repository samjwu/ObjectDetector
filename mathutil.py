"""Utility functions for calculations."""

import cv2
import numpy
import torch


def predict_transform(
    prediction: torch.Tensor,
    input_dimensions: list[int],
    anchors: list[int],
    num_classes: int,
    use_cuda: bool = True,
) -> torch.Tensor:
    """Convert a detection feature map into a 2D tensor containing bounding box attributes.

    Features are the learned representations of the input data that the model derives
    through its layers during the training process.
    Each layer of a captures different levels of abstraction.
    These intermediate representations are extracted features from the input data.

    Feature map AKA activation map
    is a mapping that corresponds to the activation of different parts of the image.
    It is a 3D representation of learned features from the input data.
    """
    batch_size = prediction.size(0)
    stride = input_dimensions // prediction.size(2)
    grid_size = input_dimensions // stride
    bounding_box_attributes = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(
        batch_size, bounding_box_attributes * num_anchors, grid_size * grid_size
    )
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size * grid_size * num_anchors, bounding_box_attributes
    )
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    grid = numpy.arange(grid_size)
    a, b = numpy.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if use_cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = (
        torch.cat((x_offset, y_offset), 1)
        .repeat(1, num_anchors)
        .view(-1, 2)
        .unsqueeze(0)
    )

    prediction[:, :, :2] += x_y_offset

    anchors = torch.FloatTensor(anchors)

    if use_cuda:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5 : 5 + num_classes] = torch.sigmoid(
        (prediction[:, :, 5 : 5 + num_classes])
    )

    prediction[:, :, :4] *= stride

    return prediction


def unique(tensor: torch.Tensor) -> torch.Tensor:
    """Filter a tensor to leave just the unique elements."""
    tensor_numpy = tensor.cpu().numpy()
    unique_numpy = numpy.unique(tensor_numpy)
    unique_tensor = torch.from_numpy(unique_numpy)

    result_tensor = tensor.new(unique_tensor.shape)
    result_tensor.copy_(unique_tensor)

    return result_tensor


def calculate_bounding_box_intersection_over_union(
    box1: torch.Tensor, box2: torch.Tensor
) -> float:
    """Calculate intersection over union of two bounding boxes."""
    # top left corner
    box1_x1, box1_y1 = box1[:, 0], box1[:, 1]
    box2_x1, box2_y1 = box2[:, 0], box2[:, 1]
    # bottom right corner
    box1_x2, box1_y2 = box1[:, 2], box1[:, 3]
    box2_x2, box2_y2 = box2[:, 2], box2[:, 3]

    # intersection rectangle coordinates
    intersection_rect_x1 = torch.max(box1_x1, box2_x1)
    intersection_rect_y1 = torch.max(box1_y1, box2_y1)
    intersection_rect_x2 = torch.min(box1_x2, box2_x2)
    intersection_rect_y2 = torch.min(box1_y2, box2_y2)

    # intersection rectangle area
    intersection_area = torch.clamp(
        intersection_rect_x2 - intersection_rect_x1 + 1, min=0
    ) * torch.clamp(intersection_rect_y2 - intersection_rect_y1 + 1, min=0)

    box1_area = (box1_x2 - box1_x1 + 1) * (box1_y2 - box1_y1 + 1)
    box2_area = (box2_x2 - box2_x1 + 1) * (box2_y2 - box2_y1 + 1)

    # intersection over union
    intersection_over_union = intersection_area / (
        box1_area + box2_area - intersection_area
    )

    return intersection_over_union


def resize_image(image: numpy.ndarray, input_dimensions: list[int]) -> numpy.ndarray:
    """Resize an image while keeping the same aspect ratio.

    Use padding to maintain the aspect ratio.
    """
    image_width = image.shape[1]
    image_height = image.shape[0]
    width, height = input_dimensions

    min_aspect_ratio = min(width / image_width, height / image_height)

    new_width = int(image_width * min_aspect_ratio)
    new_height = int(image_height * min_aspect_ratio)
    resized_image = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )

    # pad with color (128, 128, 128)
    canvas = numpy.full((input_dimensions[1], input_dimensions[0], 3), 128)

    avg_height = (height - new_height) // 2
    avg_width = (width - new_width) // 2
    canvas[
        avg_height : avg_height + new_height, avg_width : avg_width + new_width, :
    ] = resized_image

    return canvas


def prepare_image(image: numpy.ndarray, input_dimensions: list[int]) -> torch.Tensor:
    """
    Prepares an image for inputting to the neural network.

    Resizes the image while maintaining aspect ratio with padding.
    Then transposes BGR information to RGB.
    Then converts the image from numpy.ndarray to a torch.Tensor.
    """
    # loaded as BGR
    image = cv2.resize(image, (input_dimensions, input_dimensions))

    # reorder to RGB
    image = image[:, :, ::-1].transpose((2, 0, 1)).copy()

    # convert to tensor
    image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    
    return image
