import os
import json
from datetime import datetime
from typing import cast
from loader import MNIST
from convolution_layer import ConvolutionLayer
from pool_layer import PoolLayer
from softmax_layer import SoftMaxLayer
from tests.run_testing_phase import run_testing_phase
from training.predict_in_dir import predict_in_dir
from training.run_epochs import run_epochs
from utils.shuffle import shuffle

img_size = 28
samples_dir = "samples"
outdir = "out"

mndata = MNIST("./dataset")

print("Loading data...")
data_train, label_train = mndata.load_training()
data_test, label_test = mndata.load_testing()
print("Data loaded.")


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
    predict_in_dir(
        convolution_layer,
        max_pooling_layer,
        softmax_output_layer,
        samples_dir,
        outdir
    )
    exit(0)

output_classes = 10

filters_count = int(
    input("Enter the number of filters (default: 32): ").strip() or 32)
filter_size = int(input("Enter the filter size (default: 5): ").strip() or 5)
pool_size = int(input("Enter the pool size (default: 2): ").strip() or 2)

softmax_edge = int((img_size - 2) / pool_size)

num_of_epochs = int(
    input("Enter the number of epochs (default: 5): ").strip() or 5)
learning_rate = float(
    input("Enter the learning rate (default: 0.005): ").strip() or 0.005)

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
    num_of_epochs,
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

predict_in_dir(
    convolution_layer,
    max_pooling_layer,
    softmax_output_layer,
    samples_dir,
    outdir
)

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
