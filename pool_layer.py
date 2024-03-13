from typing import Generator, Tuple
import numpy as np


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

    def generate_image_regions(self, image: np.ndarray) -> Generator[
        Tuple[np.ndarray, int, int], None, None
    ]:
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
                    (row * self.filter_size):
                    (row * self.filter_size + self.filter_size),
                    (column * self.filter_size):
                    (column * self.filter_size + self.filter_size)]

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
        output: np.ndarray = np.zeros((height // self.filter_size,
                                       width // self.filter_size, num_channels))

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
                                row * self.filter_size + h, column *
                                self.filter_size + w, channel
                            ] = gradient_from_next_layer[row, column, channel]
        return gradient_input

    def serialize(self) -> dict:
        """
        Serializes the Max Pooling layer.

        Returns:
        - A dictionary containing the attributes of the Max Pooling layer.
        """
        return {
            "type": "PoolLayer",
            "filter_size": self.filter_size
        }

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
