from typing import Generator, Tuple
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
                    row : (row + self.filter_size), column : (column + self.filter_size)
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
            output[row, column] = np.sum(image_region * self.filters, axis=(1, 2))

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
