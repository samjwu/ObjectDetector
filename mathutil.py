"""Utility functions for calculations."""

import numpy
import torch


def predict_transform(
    prediction,
    input_dimensions: int,
    anchors: list[int],
    num_classes: int,
    use_cuda: bool = True,
):
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


def unique(tensor: torch.Tensor):
    """Filter a tensor to leave just the unique elements."""
    tensor_numpy = tensor.cpu().numpy()
    unique_numpy = numpy.unique(tensor_numpy)
    unique_tensor = torch.from_numpy(unique_numpy)

    result_tensor = tensor.new(unique_tensor.shape)
    result_tensor.copy_(unique_tensor)

    return result_tensor
