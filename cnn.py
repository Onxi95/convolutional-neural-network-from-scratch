import matplotlib.pyplot as plt
import numpy as np
import gzip
from loader import MNIST

mndata = MNIST("./dataset")

images, labels = mndata.load_training()

print(images[7])
print(labels[7])
image = np.asarray(images[7]).squeeze()
image = image.reshape(28, 28)

plt.imshow(image, cmap='gray')
plt.colorbar()

plt.savefig('mnist_image.png')

plt.close()
