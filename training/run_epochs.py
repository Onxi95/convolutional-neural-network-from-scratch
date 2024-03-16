import numpy as np
from convolution_layer import ConvolutionLayer
from pool_layer import PoolLayer
from softmax_layer import SoftMaxLayer
from .update_model import update_model


def run_epochs(
        train_images: np.ndarray,
        train_labels: np.ndarray,
        img_size: int,
        learning_rate: float,
        num_of_epochs: int,
        convolution_layer: ConvolutionLayer,
        max_pooling_layer: PoolLayer,
        softmax_output_layer: SoftMaxLayer
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
    print("Starting training...")
    for epoch in range(1, num_of_epochs + 1):
        print(f'Epoch {epoch}')
        cumulative_loss = 0.0
        correct_predictions = 0

        for i, (image, label) in enumerate(zip(train_images, train_labels)):
            if i % 100 == 99:  # Print progress every 100 images
                print(f'{i + 1:>5} steps')
                print(f'Training Loss {cumulative_loss / 100:.5f}')
                print(f'Training Accuracy: {correct_predictions}%')
                cumulative_loss = 0
                correct_predictions = 0

            image_array = image.reshape(img_size, img_size)
            loss, correct = update_model(
                image_array,
                label,
                learning_rate,
                convolution_layer,
                max_pooling_layer,
                softmax_output_layer
            )
            cumulative_loss += loss
            correct_predictions += correct
