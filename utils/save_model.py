import os
import json

from convolution_layer import ConvolutionLayer
from pool_layer import PoolLayer
from softmax_layer import SoftMaxLayer


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
