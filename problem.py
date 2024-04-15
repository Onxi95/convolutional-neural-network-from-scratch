# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
from __future__ import annotations
import json
import os
import sys
import logging
from datetime import datetime
from typing import cast, TYPE_CHECKING
import cv2
import numpy as np

if TYPE_CHECKING:
    from metoda import ConvolutionLayer, PoolLayer, SoftMaxLayer


# https://www.kaggle.com/datasets/zalando-research/fashionmnist
labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if not os.path.exists("logs"):
    os.makedirs("logs")

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s\n%(message)s\n",
    filename=f"logs/{now}.log",
)


def replace_and_invert(image: np.ndarray) -> np.ndarray:
    """
    Removes the white background from a grayscale image, replaces it with a black background,
    and inverts the grayscale values of the foreground.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where white regions (threshold above 240) are white
    _, binary_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Invert the mask to get black background
    inverted_mask = cv2.bitwise_not(binary_mask)

    # Apply the inverted mask to extract the foreground
    foreground = cv2.bitwise_and(gray, inverted_mask)

    # Convert 255 to a NumPy array and subtract it from the foreground
    foreground = np.subtract(np.full_like(foreground, 255), foreground)

    # Combine the inverted foreground with the black background
    result = np.where(inverted_mask == 0, 0, foreground)

    return result


def adjust_sample_image(
    source_path: str, output_path: str, shape: tuple[int, int] = (28, 28)
):
    """
    Removes the white background from an image, inverts the foreground colors to make them lighter,
    resizes the image to the specified shape, and then saves it to the specified output path.
    """

    image = cv2.imread(source_path)
    if image is None:
        logger.info("Failed to read the image from %s", source_path)
        return

    processed_image = replace_and_invert(image)
    resized_image = cv2.resize(
        processed_image, dsize=shape, interpolation=cv2.INTER_AREA
    )

    success = cv2.imwrite(output_path, resized_image)
    if not success:
        logger.info("Failed to save the image to %s", output_path)


def shuffle(data: list[int], labels: list[int], subset_size: int = 1500):
    """
    Shuffles data and labels in unison and selects a subset of the specified size.

    Parameters:
    - data (list[int]): The data to shuffle and subset.
    - labels (list[int]): The labels to shuffle in unison with the data.
    - subset_size (int): The size of the subset to return.

    Returns:
    - (numpy.ndarray, numpy.ndarray): The shuffled and subsetted data and labels.
    """

    indices = np.arange(0, np.array(data).shape[0])
    np.random.shuffle(indices)

    shuffled_data = np.array(data)[indices][:subset_size]
    shuffled_labels = np.array(labels)[indices][:subset_size]

    return shuffled_data, shuffled_labels


def save_model(
    convolution_layer: ConvolutionLayer,
    max_pooling_layer: PoolLayer,
    softmax_output_layer: SoftMaxLayer,
    model_name: str,
    model_outdir: str,
) -> None:
    """
    Saves the model to a directory.

    Args:
        convolution_layer: The convolution layer of the model.
        max_pooling_layer: The max pooling layer of the model.
        softmax_output_layer: The softmax output layer of the model.
        outdir: The directory where the model will be saved.

    Returns:
        None
    """

    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)

    with open(f"{model_outdir}/{model_name}", "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "convolutionLayer": convolution_layer.serialize(),
                    "poolLayer": max_pooling_layer.serialize(),
                    "softmaxLayer": softmax_output_layer.serialize(),
                }
            )
        )


def perform_forward_pass(
    image: np.ndarray,
    label: int,
    convolution: ConvolutionLayer,
    pool: PoolLayer,
    softMax: SoftMaxLayer,
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


def update_model(
    image: np.ndarray,
    label: int,
    learning_rate: float,
    convolution: ConvolutionLayer,
    pool: PoolLayer,
    softMax: SoftMaxLayer,
) -> tuple[float, int]:
    """
    Update the model weights based on the loss gradient, returning the loss and accuracy.
    """
    # Perform forward pass and calculate initial gradient
    softmax_probs, loss, accuracy = perform_forward_pass(
        image, label, convolution, pool, softMax
    )
    loss_gradient = np.zeros(10)
    loss_gradient[label] = -1 / softmax_probs[label]

    # Perform backpropagation through the network
    gradient_back_softmax = softMax.backward(loss_gradient, learning_rate)
    gradient_back_pool = pool.backward(gradient_back_softmax)
    convolution.backward(gradient_back_pool, learning_rate)

    return loss, accuracy


def run_epochs(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    img_size: int,
    learning_rate: float,
    num_of_epochs: int,
    convolution_layer: ConvolutionLayer,
    max_pooling_layer: PoolLayer,
    softmax_output_layer: SoftMaxLayer,
):
    """
    Runs multiple epochs of training on the given dataset.

    Args:
        train_images (np.ndarray): The training images.
        train_labels (np.ndarray): The corresponding labels for the training images.
        img_size (int): The size of the input images.
        learning_rate (float): The learning rate for the training.
        convolution_layer (ConvolutionLayer): The convolution layer of the model.
        max_pooling_layer (PoolLayer): The max pooling layer of the model.
        softmax_output_layer (SoftMaxLayer): The softmax output layer of the model.

    Returns:
        None
    """
    logger.info("Starting training...")
    for epoch in range(1, num_of_epochs + 1):
        logger.info("Epoch %s", epoch)
        cumulative_loss = 0.0
        correct_predictions = 0

        for i, (image, label) in enumerate(zip(train_images, train_labels)):
            if i % 100 == 99:  # Print progress every 100 images
                logger.info("%s steps", i + 1)
                logger.info("Training Loss %.5f", cumulative_loss / 100)
                logger.info("Training Accuracy: %s%%", correct_predictions)
                cumulative_loss = 0
                correct_predictions = 0

            image_array = image.reshape(img_size, img_size)
            loss, correct = update_model(
                image_array,
                label,
                learning_rate,
                convolution_layer,
                max_pooling_layer,
                softmax_output_layer,
            )
            cumulative_loss += loss
            correct_predictions += correct


def predict_in_dir(
    convolution_layer: ConvolutionLayer,
    max_pooling_layer: PoolLayer,
    softmax_output_layer: SoftMaxLayer,
    samples_dir: str,
    output_dir_path: str,
) -> None:
    """
    Predicts the class labels for images in a directory using a given model.

    Args:
        model: The trained model used for prediction.
        dir_path: The path to the directory containing the input images.
        output_dir_path: The path to the directory where the predicted images will be saved.

    Returns:
        None
    """

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    samples = os.listdir(samples_dir)

    for sample in samples:
        source_path = os.path.join(samples_dir, sample)
        output_path = os.path.join(output_dir_path, sample)
        logger.info("%s -> %s", source_path, output_path)
        adjust_sample_image(source_path, output_path)

    if os.path.exists(output_dir_path):
        adjusted_samples = os.listdir(output_dir_path)

        for sample in adjusted_samples:
            sample_image = cv2.imread(
                f"./{output_dir_path}/{sample}", cv2.IMREAD_GRAYSCALE
            )
            softmax_probs, _, _ = perform_forward_pass(
                sample_image,
                0,
                convolution_layer,
                max_pooling_layer,
                softmax_output_layer,
            )

            prediction = labels[np.argmax(softmax_probs)]
            top_3_guesses = list(
                map(
                    lambda x: (labels[x[0]], x[1]),
                    sorted(enumerate(softmax_probs), key=lambda x: x[1], reverse=True)[
                        :3
                    ],
                )
            )
            logger.info(
                "Prediction for %s: %s, probs: %s", sample, prediction, top_3_guesses
            )


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
