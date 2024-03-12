import cv2
import numpy as np


def replace_and_invert(image: np.ndarray) -> np.ndarray:
    """
    Removes the white background from an image, replaces it with a black background, 
    and inverts the colors of the foreground.
    """
    # Convert image to grayscale to detect the white background.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Create an inverted mask for the foreground.
    inverted_mask = cv2.bitwise_not(binary_mask)

    # Extract the foreground by applying the mask.
    foreground = cv2.bitwise_and(image, image, mask=inverted_mask)

    # Determine where the foreground is in the image.
    foreground_mask = np.any(foreground != [0, 0, 0], axis=-1)

    # Prepare an empty (black) background.
    background = np.zeros_like(image)

    # Combine the extracted foreground with the empty background.
    combined = np.where(foreground == 0, background, foreground)

    # Invert the colors of the foreground.
    combined[foreground_mask] = 255 - combined[foreground_mask]

    return combined


def adjust_sample_image(source_path: str, output_path: str, shape: tuple[int, int] = (28, 28)):
    """
    Removes the white background from an image, inverts the foreground colors to make them lighter,
    resizes the image to the specified shape, and then saves it to the specified output path.
    """

    image = cv2.imread(source_path)
    if image is None:
        print(f"Failed to read the image from {source_path}")
        return

    processed_image = replace_and_invert(image)
    resized_image = cv2.resize(
        processed_image, dsize=shape, interpolation=cv2.INTER_AREA)

    success = cv2.imwrite(output_path, resized_image)
    if not success:
        print(f"Failed to save the image to {output_path}")
