import numpy as np

from convolution_layer import ConvolutionLayer
from pool_layer import PoolLayer
from softmax_layer import SoftMaxLayer
from training.perform_forward_pass import perform_forward_pass
from utils.labels import labels
from utils.logger import logger


def run_testing_phase(
    test_images: np.ndarray,
    test_labels: np.ndarray,
    img_size: int,
    convolution_layer: ConvolutionLayer,
    max_pooling_layer: PoolLayer,
    softmax_output_layer: SoftMaxLayer,
):
    """
    Args:
        test_images (np.ndarray): The array of test images.
        test_labels (np.ndarray): The array of corresponding labels for the test images.
        img_size (int): The size of the images (assumed to be square).
        convolution_layer: The convolution layer of the neural network.
        max_pooling_layer: The max pooling layer of the neural network.
        softmax_output_layer: The softmax output layer of the neural network.

    Returns:
        tuple: A tuple containing the test loss and test accuracy.

    """
    confusion_matrix = np.zeros((10, 10), dtype=int)

    logger.info("Testing phase...")
    total_loss = 0
    total_correct = 0

    for image, label in zip(test_images, test_labels):
        image_array = image.reshape(img_size, img_size)
        softmax_probs, loss, correct = perform_forward_pass(
            image_array,
            label,
            convolution_layer,
            max_pooling_layer,
            softmax_output_layer,
        )
        total_loss += loss
        total_correct += correct
        predicted_label = np.argmax(softmax_probs)
        confusion_matrix[label, predicted_label] += 1

    num_tests = len(test_images)
    test_loss = total_loss / num_tests
    test_accuracy = total_correct / num_tests
    logger.info("Test Loss: %s", test_loss)
    logger.info("Test Accuracy: %.2f%%", test_accuracy * 100)

    logger.info("Confusion Matrix:")
    print_confusion_matrix_with_labels(confusion_matrix, list(labels.values()))

    return test_loss, test_accuracy


def print_confusion_matrix_with_labels(
    confusion_matrix, labels: list[str], column_width=12
):
    """
    Prints the confusion matrix with class labels.
    """
    header = (
        " " * column_width
        + "| "
        + " | ".join([label.center(column_width) for label in labels])
    )
    separator = "-" * len(header)

    table = separator + "\n" + header + "\n" + separator + "\n"

    for i, row_label in enumerate(labels):
        row_str = f"{row_label.center(column_width)}| "
        for value in confusion_matrix[i]:
            # row percentage
            percentage = (
                (value / sum(confusion_matrix[i])) * 100
                if sum(confusion_matrix[i]) != 0
                else 0
            )
            value_str = f"{value} ({percentage:.0f}%)".center(column_width)
            row_str += value_str + " | "
        table += row_str + "\n" + separator + "\n"

    logger.info(table)
