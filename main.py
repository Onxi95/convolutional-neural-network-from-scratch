import os
import json
from datetime import datetime
from typing import cast
import numpy as np
import cv2
from loader import MNIST
from convolution_layer import ConvolutionLayer
from pool_layer import PoolLayer
from softmax_layer import SoftMaxLayer
from tests.run_testing_phase import run_testing_phase
from training.perform_forward_pass import perform_forward_pass
from training.run_epochs import run_epochs
from utils.shuffle import shuffle
from utils.adjust_sample_image import adjust_sample_image


mndata = MNIST("./dataset")

print("Loading data...")
data_train, label_train = mndata.load_training()
data_test, label_test = mndata.load_testing()
print("Data loaded.")

img_size = 28
filters_count = 64
filter_size = 3
pool_size = 1
softmax_edge = int((img_size - 2) / pool_size)
output_classes = 10
num_of_epochs = 4

learning_rate = 0.01

print("Shuffling data...")

train_images, train_labels = shuffle(
    cast(list[int], data_train), cast(list[int], label_train))
test_images, test_labels = shuffle(
    cast(list[int], data_test), cast(list[int], label_test))

print("Data shuffled.")

should_train_again = input("Do you want to train the model? (Y/n): ")
if should_train_again.lower() == "n":
    print("Skipping training...")
    path: str = input("Enter the relative path of the model to load: ")
    print("Loading model...")
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    print("Model loaded.")
    print("Initializing layers...")
    convolution_layer = ConvolutionLayer.deserialize(model)
    max_pooling_layer = PoolLayer.deserialize(model)
    softmax_output_layer = SoftMaxLayer.deserialize(model)
    print("Layers initialized.")
    run_testing_phase(
        test_images,
        test_labels,
        img_size,
        convolution_layer,
        max_pooling_layer,
        softmax_output_layer
    )
    exit(0)

print("Initializing layers...")
convolution_layer = ConvolutionLayer(filters_count, filter_size)
max_pooling_layer = PoolLayer(pool_size)
softmax_output_layer = SoftMaxLayer(
    softmax_edge**2 * filters_count, output_classes)
print("Layers initialized.")

run_epochs(
    train_images,
    train_labels,
    img_size,
    learning_rate,
    convolution_layer,
    max_pooling_layer,
    softmax_output_layer
)

run_testing_phase(
    test_images,
    test_labels,
    img_size,
    convolution_layer,
    max_pooling_layer,
    softmax_output_layer
)

samplesdir = "samples"
outdir = "out"
samples = os.listdir(samplesdir)

if not os.path.exists(outdir):
    os.makedirs(outdir)

for sample in samples:
    source_path = os.path.join(samplesdir, sample)
    output_path = os.path.join(outdir, sample)
    print(f"{source_path} -> {output_path}")
    adjust_sample_image(source_path, output_path)

if os.path.exists(outdir):
    adjusted_samples = os.listdir(outdir)

    for sample in adjusted_samples:
        sample_image = cv2.imread(f'./{outdir}/{sample}', cv2.IMREAD_GRAYSCALE)
        softmax_probs, _, _ = perform_forward_pass(
            sample_image, 0, convolution_layer, max_pooling_layer, softmax_output_layer
        )
        print(
            f'Prediction for {sample}: {np.argmax(softmax_probs)}, probs: {softmax_probs}')

model_outdir = "model"
if not os.path.exists(model_outdir):
    os.makedirs(model_outdir)

current_date = datetime.today().strftime('%Y-%m-%d-%H:%m')

with open(f'{model_outdir}/model-{current_date}.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(
        {
            "convolutionLayer": convolution_layer.serialize(),
            "poolLayer": max_pooling_layer.serialize(),
            "softmaxLayer": softmax_output_layer.serialize(),
        }
    ))
