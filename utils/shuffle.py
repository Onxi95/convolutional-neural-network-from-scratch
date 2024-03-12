import numpy as np


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

    indices = np.arange(np.array(data).shape[0])
    np.random.shuffle(indices)

    shuffled_data = np.array(data)[indices][:subset_size]
    shuffled_labels = np.array(labels)[indices][:subset_size]

    return shuffled_data, shuffled_labels
