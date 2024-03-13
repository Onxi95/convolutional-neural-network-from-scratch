from typing import cast
import numpy as np
from convolution_layer import ConvolutionLayer
from pool_layer import PoolLayer
from softmax_layer import SoftMaxLayer


def perform_forward_pass(
    image: np.ndarray, label: int,
    convolution: ConvolutionLayer,
    pool: PoolLayer,
    softMax: SoftMaxLayer
) -> tuple[np.ndarray, float, int]:
    """
    Perform a forward pass through the CNN, calculate the loss and accuracy.
    """
    # scales the pixel values of the image to the range [0, 1]
    # and shifts the range from [0, 1] to [-0.5, 0.5]
    normalized_image = (image / 255.0) - 0.5
    convolution_output = convolution.forward(normalized_image)
    pooled_output = pool.forward(convolution_output)
    softmax_probs = softMax.forward(pooled_output)

    loss = cast(float, -np.log(softmax_probs[label]))
    accuracy = int(np.argmax(softmax_probs) == label)

    return softmax_probs, loss, accuracy
