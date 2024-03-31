# https://github.com/sorki/python-mnist/blob/master/mnist/loader.py

import os
import struct
from array import array


class MNIST(object):
    """
    MNIST class for loading and handling MNIST dataset.

    Attributes:
        path (str): The path to the directory containing the MNIST dataset files.
        test_img_fname (str): The filename of the testing images file.
        test_lbl_fname (str): The filename of the testing labels file.
        train_img_fname (str): The filename of the training images file.
        train_lbl_fname (str): The filename of the training labels file.
        test_images (list): The loaded testing images.
        test_labels (list): The loaded testing labels.
        train_images (list): The loaded training images.
        train_labels (list): The loaded training labels.
    """

    def __init__(self, path="."):
        """
        Initialize the MNIST object.

        Args:
            path (str, optional): The path to the directory containing the MNIST dataset files.
                Defaults to '.'.
        """
        self.path = path

        self.test_img_fname = "t10k-images-idx3-ubyte"
        self.test_lbl_fname = "t10k-labels-idx1-ubyte"

        self.train_img_fname = "train-images-idx3-ubyte"
        self.train_lbl_fname = "train-labels-idx1-ubyte"

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        """
        Load the testing images and labels from the specified file paths.

        Returns:
            A tuple containing the loaded testing images and labels.
        """
        ims, labels = self.load(
            os.path.join(self.path, self.test_img_fname),
            os.path.join(self.path, self.test_lbl_fname),
        )

        self.test_images = ims
        self.test_labels = labels

        return self.test_images, self.test_labels

    def load_training(self):
        """
        Loads the training images and labels from the specified file paths.

        Returns:
            tuple: A tuple containing the loaded training images and labels.
        """
        ims, labels = self.load(
            os.path.join(self.path, self.train_img_fname),
            os.path.join(self.path, self.train_lbl_fname),
        )

        self.train_images = ims
        self.train_labels = labels

        return self.train_images, self.train_labels

    def load(self, path_img, path_lbl, batch=None):
        """
        Load image and label data from the specified paths.

        Args:
            path_img (str): The path to the image data file.
            path_lbl (str): The path to the label data file.
            batch (tuple, optional): A tuple containing the start index and the number of samples to load.
                                    If specified, only a subset of the data will be loaded.
                                    Defaults to None, which loads all the data.

        Returns:
            tuple: A tuple containing the loaded images and labels.

        Raises:
            ValueError: If the magic number in the label data file does not match
              the expected value (2049).
            ValueError: If the magic number in the image data file does not match
              the expected value (2051).

        """
        with open(path_lbl, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")

            labels = array("B", file.read())

        with open(path_img, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")

            image_data = array("B", file.read())

        if batch is not None:
            image_data = image_data[
                batch[0] * rows * cols : (batch[0] + batch[1]) * rows * cols
            ]
            labels = labels[batch[0] : batch[0] + batch[1]]
            size = batch[1]

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols : (i + 1) * rows * cols]

        return images, labels
