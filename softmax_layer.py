from typing import Tuple
import numpy as np


class SoftMaxLayer:
    """
    It converts logits (the outputs of the previous layers) into probabilities by applying the
    softmax function. Each neuron in the layer represents a class, and the output of each neuron
    is the probability that the given input belongs to that class.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initializes the Softmax layer with random weights and zero biases.

        Parameters:
        - input_size: The number of input nodes.
        - output_size: The number of output nodes (classes).
        """
        self.weights: np.ndarray = np.random.randn(
            input_size, output_size) / input_size
        self.biases: np.ndarray = np.zeros(output_size)
        self.original_input_shape: Tuple[int, ...] = ()
        self.flattened_input: np.ndarray = np.array([])
        self.output_logits: np.ndarray = np.array([])

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the softmax layer.

        Parameters:
        - input_tensor: The input tensor, typically an image flattened into a 1D array.

        Returns:
        - The probabilities of each class.
        """

        # Preserve original shape for backprop
        self.original_input_shape: Tuple[int, ...] = input_tensor.shape
        input_flattened: np.ndarray = input_tensor.flatten()
        self.flattened_input: np.ndarray = input_flattened  # Stored for backpropagation

        logits: np.ndarray = np.dot(
            input_flattened, self.weights) + self.biases
        self.output_logits: np.ndarray = logits  # Stored for backpropagation

        exp_logits: np.ndarray = np.exp(logits)
        probabilities: np.ndarray = exp_logits / np.sum(exp_logits, axis=0)

        return probabilities

    def backward(self, gradient_of_loss: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Performs the backward pass of the softmax layer, updating weights 
        and biases based on the gradient of the loss.

        Parameters:
        - gradient_of_loss: The gradient of the loss with respect to the output of this layer.
        - learning_rate: The learning rate to use for the updates.

        Returns:
        - The gradient of the loss with respect to the input of this layer.
        """

        for i, gradient in enumerate(gradient_of_loss):
            if gradient == 0:
                continue

            exp_logits: np.ndarray = np.exp(self.output_logits)
            sum_exp_logits: float = np.sum(exp_logits)

            gradient_softmax_output = - \
                exp_logits[i] * exp_logits / (sum_exp_logits ** 2)
            gradient_softmax_output[i] = exp_logits[i] * \
                (sum_exp_logits - exp_logits[i]) / (sum_exp_logits ** 2)

            gradient_logits_weights = self.flattened_input
            gradient_logits_biases = 1
            gradient_logits_input = self.weights

            gradient_loss_logits = gradient * gradient_softmax_output

            gradient_loss_weights = np.outer(
                gradient_logits_weights, gradient_loss_logits)
            gradient_loss_biases = gradient_loss_logits * gradient_logits_biases
            gradient_loss_input = gradient_logits_input.dot(
                gradient_loss_logits)

            self.weights -= learning_rate * gradient_loss_weights
            self.biases -= learning_rate * gradient_loss_biases

            return gradient_loss_input.reshape(self.original_input_shape)

        return np.zeros(self.original_input_shape)
