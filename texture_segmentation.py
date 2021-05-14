import cv2
from math import log, e
import numpy as np
from scipy.stats import entropy


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


def get_entropy(window, base=None):
    value, counts = np.unique(window, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


# TODO: check what is faster
def get_entropy2(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def get_sliding_window_properties(input_image):
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    #pure_image = image[120:220, 150:300]
    pure_image = image[0:400, 0:500]
    (winW, winH) = (4, 4)

    variance_matrix = np.zeros(pure_image.shape, np.uint8)  # (100, 150, 3)
    std_matrix = np.zeros(pure_image.shape, np.uint8)
    mean_matrix = np.zeros(pure_image.shape, np.uint8)
    entropy_matrix = np.zeros(pure_image.shape, np.uint8)
    for x, y, window in sliding_window(
            pure_image, stepSize=1, windowSize=(winW, winH)
    ):
        variance_matrix[y][x] = int(np.var(window))  # дисперсия
        std_matrix[y][x] = np.std(window)  # СКО
        mean_matrix[y][x] = np.mean(window)  # среднее
        entropy_matrix[y][x] = get_entropy(window)  # энтропия
    # show results
    # multiply by 2 to make brighter as an example
    cv2.imshow('variance', 2 * entropy_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


input_img = 'images/emulsion/Bayer12p_Image__2021-05-06__13-12-35.tiff'
get_sliding_window_properties('images/test.png')
