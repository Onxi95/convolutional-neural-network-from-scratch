import matplotlib.pyplot as plt
import numpy as np
import gzip
from loader import MNIST
from convolution_layer import ConvolutionLayer
from pool_layer import PoolLayer
from softmax_layer import SoftMaxLayer

mndata = MNIST("./dataset")

data_train, label_train = mndata.load_training()
data_test, label_test = mndata.load_testing()

img_size = 28
filters_count = 8
filter_size = 3
pool_size = 1
softmax_edge = int((img_size - 2) / pool_size)
output_classes = 10

convolution_layer = ConvolutionLayer(filters_count, filter_size)
max_pooling_layer = PoolLayer(pool_size)
softmax_output_layer = SoftMaxLayer(
    softmax_edge**2 * filters_count, output_classes)

# Sample a subset for training and testing
indices = np.arange(np.array(data_train).shape[0])
np.random.shuffle(indices)

subset_size = 1500
train_images = data_train[:subset_size]
train_labels = label_train[:subset_size]
test_images = data_test[:subset_size]
test_labels = label_test[:subset_size]


def perform_forward_pass(image, label):
    """
    Perform a forward pass through the CNN, calculate the loss and accuracy.
    """
    # scales the pixel values of the image to the range [0, 1]
    # and shifts the range from [0, 1] to [-0.5, 0.5]
    normalized_image = (image / 255.0) - 0.5
    convolution_output = convolution_layer.forward(normalized_image)
    pooled_output = max_pooling_layer.forward(convolution_output)
    softmax_probs = softmax_output_layer.forward(pooled_output)

    loss = -np.log(softmax_probs[label])
    accuracy = int(np.argmax(softmax_probs) == label)

    return softmax_probs, loss, accuracy


def update_model(image: np.ndarray, label: int, learning_rate: float = 0.003) -> (float, int):
    """
    Update the model weights based on the loss gradient, returning the loss and accuracy.
    """
    # Perform forward pass and calculate initial gradient
    softmax_probs, loss, accuracy = perform_forward_pass(image, label)
    loss_gradient = np.zeros(10)
    loss_gradient[label] = -1 / softmax_probs[label]

    # Perform backpropagation through the network
    gradient_back_softmax = softmax_output_layer.backward(
        loss_gradient, learning_rate)
    gradient_back_pool = max_pooling_layer.backward(gradient_back_softmax)
    convolution_layer.backward(gradient_back_pool, learning_rate)

    return loss, accuracy


print("Starting training...")
for epoch in range(1, 5):
    print(f'Epoch {epoch}')
    cumulative_loss = 0.0
    correct_predictions = 0

    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 99:  # Print progress every 100 images
            print(
                f'{i + 1:>5} steps: Training Loss {cumulative_loss / 100:.5f} | Training Accuracy: {correct_predictions}%')
            cumulative_loss = 0
            correct_predictions = 0

        image_array = np.array(image).reshape(img_size, img_size)
        loss, correct = update_model(image_array, label)
        cumulative_loss += loss
        correct_predictions += correct

print("Testing phase...")
total_loss = 0
total_correct = 0

for image, label in zip(test_images, test_labels):
    image_array = np.array(image).reshape(img_size, img_size)
    _, loss, correct = perform_forward_pass(image_array, label)
    total_loss += loss
    total_correct += correct

num_tests = len(test_images)
print(f'Test Loss: {total_loss / num_tests}')
print(f'Test Accuracy: {total_correct / num_tests}')
