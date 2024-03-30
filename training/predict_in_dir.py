import os
import cv2
import numpy as np

from convolution_layer import ConvolutionLayer
from pool_layer import PoolLayer
from softmax_layer import SoftMaxLayer
from training.perform_forward_pass import perform_forward_pass
from utils.adjust_sample_image import adjust_sample_image
from utils.labels import labels
from utils.logger import logger


def predict_in_dir(convolution_layer: ConvolutionLayer,
                   max_pooling_layer: PoolLayer,
                   softmax_output_layer: SoftMaxLayer,
                   samples_dir: str,
                   output_dir_path: str
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
                f'./{output_dir_path}/{sample}', cv2.IMREAD_GRAYSCALE)
            softmax_probs, _, _ = perform_forward_pass(
                sample_image, 0, convolution_layer, max_pooling_layer, softmax_output_layer
            )

            prediction = labels[np.argmax(softmax_probs)]
            top_3_guesses = list(map(
                lambda x: (labels[x[0]], x[1]),
                sorted(
                    enumerate(softmax_probs),
                    key=lambda x: x[1], reverse=True)[:3]
            ))
            logger.info(
                'Prediction for %s: %s, probs: %s', sample, prediction, top_3_guesses)
