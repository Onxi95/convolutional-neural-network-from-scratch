import cv2
import numpy as np
from logger import logger


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
