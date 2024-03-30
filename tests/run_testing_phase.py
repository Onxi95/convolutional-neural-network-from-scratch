import numpy as np

from convolution_layer import ConvolutionLayer
from pool_layer import PoolLayer
from softmax_layer import SoftMaxLayer
from training.perform_forward_pass import perform_forward_pass


def run_testing_phase(
        test_images: np.ndarray,
        test_labels: np.ndarray,
        img_size: int,
        convolution_layer: ConvolutionLayer,
        max_pooling_layer: PoolLayer,
        softmax_output_layer: SoftMaxLayer
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

    print("Testing phase...")
    total_loss = 0
    total_correct = 0

    for image, label in zip(test_images, test_labels):
        image_array = image.reshape(img_size, img_size)
        softmax_probs, loss, correct = perform_forward_pass(
            image_array, label, convolution_layer, max_pooling_layer, softmax_output_layer)
        total_loss += loss
        total_correct += correct
        predicted_label = np.argmax(softmax_probs)
        confusion_matrix[label, predicted_label] += 1

    num_tests = len(test_images)
    test_loss = total_loss / num_tests
    test_accuracy = total_correct / num_tests
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    print("Confusion Matrix:")
    print(confusion_matrix)

    return test_loss, test_accuracy
