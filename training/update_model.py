import numpy as np
from convolution_layer import ConvolutionLayer
from pool_layer import PoolLayer
from softmax_layer import SoftMaxLayer
from .perform_forward_pass import perform_forward_pass


def update_model(
        image: np.ndarray, label: int,
        learning_rate: float,
        convolution: ConvolutionLayer,
        pool: PoolLayer,
        softMax: SoftMaxLayer) -> tuple[float, int]:
    """
    Update the model weights based on the loss gradient, returning the loss and accuracy.
    """
    # Perform forward pass and calculate initial gradient
    softmax_probs, loss, accuracy = perform_forward_pass(
        image, label, convolution, pool, softMax)
    loss_gradient = np.zeros(10)
    loss_gradient[label] = -1 / softmax_probs[label]

    # Perform backpropagation through the network
    gradient_back_softmax = softMax.backward(
        loss_gradient, learning_rate
    )
    gradient_back_pool = pool.backward(gradient_back_softmax)
    convolution.backward(gradient_back_pool, learning_rate)

    return loss, accuracy
