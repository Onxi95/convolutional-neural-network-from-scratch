from typing import Tuple, Generator
import numpy as np


class ConvolutionLayer:
    """
    A class representing a Convolution layer in a neural network, including methods for
    forward propagation and back propagation for weight updates.
    """

    def __init__(self, number_of_filters: int, filter_size: int) -> None:
        """
        Initializes the Convolution layer object.

        Parameters:
        - number_of_filters: The number of filters in the convolution layer.
        - filter_size: The dimension of each square filter (filter_size x filter_size).
        """

        self.number_of_filters: int = number_of_filters
        self.filter_size: int = filter_size
        # Initialize filters as a 3D array of random values, normalized by the filter size squared.
        self.filters: np.ndarray = np.random.randn(
            number_of_filters, filter_size, filter_size
        ) / (filter_size**2)

    def generate_image_regions(
        self, image: np.ndarray
    ) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        A generator function that yields regions of the image to be convolved with filters,
        including the region's top-left corner coordinates.

        Parameters:
        - image: The input image as a 2D array.
        """

        image_height, image_width = image.shape
        self.image: np.ndarray = image

        for row in range(image_height - self.filter_size + 1):
            for column in range(image_width - self.filter_size + 1):
                image_region = image[
                    row: (row + self.filter_size), column: (column + self.filter_size)
                ]
                yield image_region, row, column

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Performs the forward propagation by applying the convolution operation between
        the image and filters.

        Parameters:
        - image: The input image as a 2D array.

        Returns:
        - output: The output from the convolution operation.
        """

        image_height, image_width = image.shape
        output_height = image_height - self.filter_size + 1
        output_width = image_width - self.filter_size + 1

        # Initialize the convolution output as a 3D array.
        output: np.ndarray = np.zeros(
            (output_height, output_width, self.number_of_filters)
        )
        for image_region, row, column in self.generate_image_regions(image):
            output[row, column] = np.sum(
                image_region * self.filters, axis=(1, 2))

        return output

    def backward(self, loss_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Updates the filters during back propagation based on the gradient of the loss function.

        Parameters:
        - loss_gradient: The gradient of the loss with respect to this layer's output.
        - learning_rate: The learning rate used for updating the filter weights.

        Returns:
        - filters_gradient: The gradient of the loss function with respect to the filter parameters.
        """

        filters_gradient: np.ndarray = np.zeros(self.filters.shape)
        for image_region, row, column in self.generate_image_regions(self.image):
            for filter_index in range(self.number_of_filters):
                filters_gradient[filter_index] += (
                    image_region * loss_gradient[row, column, filter_index]
                )

        # Update filters by subtracting a portion of the gradient determined by the learning rate.
        self.filters -= learning_rate * filters_gradient
        return filters_gradient

    def serialize(self) -> dict:
        """
        Returns a dictionary of the Convolution layer's attributes for serialization.
        """

        layer_dict = {
            "type": "ConvolutionLayer",
            "number_of_filters": self.number_of_filters,
            "filter_size": self.filter_size,
            "filters": self.filters.tolist(),
        }
        return layer_dict

    @staticmethod
    def deserialize(model: dict):
        """
        Deserializes a dictionary to a Convolution layer object.

        Parameters:
        - model: A dictionary containing the attributes of the Convolution layer.

        Returns:
        - A Convolution layer object.

        """
        layer = model["convolutionLayer"]
        number_of_filters = layer["number_of_filters"]
        filter_size = layer["filter_size"]
        filters = layer["filters"]

        Convolution = ConvolutionLayer(number_of_filters, filter_size)
        Convolution.filters = np.array(filters)

        return Convolution


class PoolLayer:
    """
    Implements a Max Pooling layer for a neural network that reduces the spatial dimensions
    (height, width) of the input image by taking the maximum value over a specified window
    for each channel independently, effectively downsampling the image.
    """

    def __init__(self, filter_size: int) -> None:
        """
        Initializes the Max Pooling layer with a specified square filter size.

        Parameters:
        - filter_size: Size of the window over which to take the maximum, for both width and height.
        """
        self.filter_size: int = filter_size
        self.image: np.ndarray = np.array([])

    def generate_image_regions(
        self, image: np.ndarray
    ) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        Yields square regions of the image based on the filter size along with their
        top-left corner coordinates.

        Parameters:
        - image: The input image as a 3D array (height, width, channels).
        """
        new_height: int = image.shape[0] // self.filter_size
        new_width: int = image.shape[1] // self.filter_size
        self.image: np.ndarray = image

        for row in range(new_height):
            for column in range(new_width):
                image_region = image[
                    (row * self.filter_size): (
                        row * self.filter_size + self.filter_size
                    ),
                    (column * self.filter_size): (
                        column * self.filter_size + self.filter_size
                    ),
                ]

                yield image_region, row, column

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Max Pooling operation to the input image.

        Parameters:
        - image: Input image as a 3D array (height, width, channels).

        Returns:
        - Pooled output as a 3D array.
        """
        height, width, num_channels = image.shape
        output: np.ndarray = np.zeros(
            (height // self.filter_size, width // self.filter_size, num_channels)
        )

        for image_region, row, column in self.generate_image_regions(image):
            output[row, column] = np.amax(image_region, axis=(0, 1))

        return output

    def backward(self, gradient_from_next_layer: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through the Max Pooling layer.

        Parameters:
        - gradient_from_next_layer: Gradient of the loss with respect to the output of this layer.

        Returns:
        - Gradient of the loss with respect to the input of this layer.
        """
        gradient_input: np.ndarray = np.zeros(self.image.shape)
        for image_region, row, column in self.generate_image_regions(self.image):
            height, width, num_channels = image_region.shape
            max_values = np.amax(image_region, axis=(0, 1))

            for h in range(height):
                for w in range(width):
                    for channel in range(num_channels):
                        # Pass gradient only to the maximum value within each pooling region.
                        if image_region[h, w, channel] == max_values[channel]:
                            gradient_input[
                                row * self.filter_size + h,
                                column * self.filter_size + w,
                                channel,
                            ] = gradient_from_next_layer[row, column, channel]
        return gradient_input

    def serialize(self) -> dict:
        """
        Serializes the Max Pooling layer.

        Returns:
        - A dictionary containing the attributes of the Max Pooling layer.
        """
        return {"type": "PoolLayer", "filter_size": self.filter_size}

    @staticmethod
    def deserialize(data: dict):
        """
        Deserializes the Max Pooling layer from a dictionary.

        Parameters:
        - data: A dictionary containing the attributes of the Max Pooling layer.

        Returns:
        - A new PoolLayer instance.
        """
        filter_size = data["poolLayer"]["filter_size"]

        Pool = PoolLayer(filter_size)
        return Pool


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
        self.input_size = input_size
        self.output_size = output_size
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

    def backward(
        self, gradient_of_loss: np.ndarray, learning_rate: float
    ) -> np.ndarray:
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
                exp_logits[i] * exp_logits / (sum_exp_logits**2)
            gradient_softmax_output[i] = (
                exp_logits[i] * (sum_exp_logits - exp_logits[i]
                                 ) / (sum_exp_logits**2)
            )

            gradient_logits_weights = self.flattened_input
            gradient_logits_biases = 1
            gradient_logits_input = self.weights

            gradient_loss_logits = gradient * gradient_softmax_output

            gradient_loss_weights = np.outer(
                gradient_logits_weights, gradient_loss_logits
            )
            gradient_loss_biases = gradient_loss_logits * gradient_logits_biases
            gradient_loss_input = gradient_logits_input.dot(
                gradient_loss_logits)

            self.weights -= learning_rate * gradient_loss_weights
            self.biases -= learning_rate * gradient_loss_biases

            return gradient_loss_input.reshape(self.original_input_shape)

        return np.zeros(self.original_input_shape)

    def serialize(self) -> dict:
        """
        Serializes the layer to a dictionary of its weights and biases.

        Returns:
        - A dictionary containing the layer's weights and biases.
        """
        return {
            "type": "SoftmaxLayer",
            "input_size": self.input_size,
            "output_size": self.output_size,
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
        }

    @staticmethod
    def deserialize(data: dict):
        """
        Deserializes the layer from a dictionary.

        Parameters:
        - data: A dictionary containing the layer's input size, output size, weights and biases.

        Returns:
        - A new SoftmaxLayer instance.
        """

        layer = data["softmaxLayer"]
        input_size = layer["input_size"]
        output_size = layer["output_size"]
        weights = np.array(layer["weights"])
        biases = np.array(layer["biases"])

        SoftMax = SoftMaxLayer(input_size, output_size)
        SoftMax.weights = weights
        SoftMax.biases = biases
        return SoftMax
