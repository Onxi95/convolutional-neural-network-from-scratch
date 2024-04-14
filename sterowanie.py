import os
import json
from datetime import datetime
import struct
from typing import cast
from array import array
from metoda import ConvolutionLayer, PoolLayer, SoftMaxLayer
from problem import shuffle, logger, save_model, run_epochs, predict_in_dir, run_testing_phase


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
                raise ValueError(
                    f"Magic number mismatch, expected 2049, got {magic}")

            labels = array("B", file.read())

        with open(path_img, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    f"Magic number mismatch, expected 2051, got {magic}")

            image_data = array("B", file.read())

        if batch is not None:
            image_data = image_data[
                batch[0] * rows * cols: (batch[0] + batch[1]) * rows * cols
            ]
            labels = labels[batch[0]: batch[0] + batch[1]]
            size = batch[1]

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols: (i + 1) * rows * cols]

        return images, labels


img_size = 28
samples_dir = "samples"
outdir = "out"

mndata = MNIST("./dataset")

logger.info("Loading data...")
data_train, label_train = mndata.load_training()
data_test, label_test = mndata.load_testing()
logger.info("Data loaded.")

training_set_size = int(
    input("Enter the size of the training set (default: 10 000, max: 60 000): ")
    or 10_000
)
logger.info("Training set size: %s", training_set_size)
test_set_size = int(
    input("Enter the size of the training set (default: 1 000, max: 10 000): ") or 1_000
)
logger.info("Test set size: %s", test_set_size)

logger.info("Shuffling data...")

train_images, train_labels = shuffle(
    cast(list[int], data_train), cast(
        list[int], label_train), training_set_size
)
test_images, test_labels = shuffle(
    cast(list[int], data_test), cast(list[int], label_test), test_set_size
)

logger.info("Data shuffled.")

should_train_again = input("Do you want to train the model? (Y/n): ")
if should_train_again.lower() == "n":
    logger.info("Skipping training...")
    path: str = input("Enter the relative path of the model to load: ")
    logger.info("Loading model...")
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    logger.info("Model loaded.")
    logger.info("Initializing layers...")
    convolution_layer = ConvolutionLayer.deserialize(model)
    max_pooling_layer = PoolLayer.deserialize(model)
    softmax_output_layer = SoftMaxLayer.deserialize(model)
    logger.info("Layers initialized.")
    run_testing_phase(
        test_images,
        test_labels,
        img_size,
        convolution_layer,
        max_pooling_layer,
        softmax_output_layer,
    )
    predict_in_dir(
        convolution_layer, max_pooling_layer, softmax_output_layer, samples_dir, outdir
    )
    exit(0)

output_classes = 10
filter_size = 3

filters_count = int(
    input("Enter the number of filters (default: 32): ").strip() or 32)
logger.info("Number of filters: %s", filters_count)
pool_size = int(input("Enter the pool size (default: 1): ").strip() or 1)
logger.info("Pool size: %s", pool_size)
softmax_edge = int((img_size - 2) / pool_size)

num_of_epochs = int(
    input("Enter the number of epochs (default: 5): ").strip() or 5)
logger.info("Number of epochs: %s", num_of_epochs)
learning_rate = float(
    input("Enter the learning rate (default: 0.005): ").strip() or 0.005
)
logger.info("Learning rate: %s", learning_rate)

logger.info("Initializing layers...")
convolution_layer = ConvolutionLayer(filters_count, filter_size)
max_pooling_layer = PoolLayer(pool_size)
softmax_output_layer = SoftMaxLayer(
    softmax_edge**2 * filters_count, output_classes)
logger.info("Layers initialized.")

run_epochs(
    train_images,
    train_labels,
    img_size,
    learning_rate,
    num_of_epochs,
    convolution_layer,
    max_pooling_layer,
    softmax_output_layer,
)

run_testing_phase(
    test_images,
    test_labels,
    img_size,
    convolution_layer,
    max_pooling_layer,
    softmax_output_layer,
)

predict_in_dir(
    convolution_layer, max_pooling_layer, softmax_output_layer, samples_dir, outdir
)

model_outdir = (
    input("Enter the directory to save the model (default: 'model'): ") or "model"
)

if not os.path.exists(model_outdir):
    os.makedirs(model_outdir)

current_date = datetime.today().strftime("%Y-%m-%d-%H:%m")
model_name = (
    input(f"Enter the name of the model (default: {current_date}.json): ")
    or current_date
)

save_model(
    convolution_layer, max_pooling_layer, softmax_output_layer, model_name, model_outdir
)
