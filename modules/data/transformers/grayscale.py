import os
import sys
sys.path.append('..')

import cv2 as cv

import resources.utils as utils


def transform_function(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def test_transformer():
    transformer = utils.CervTransformer(name="Gray Scaling", transfunc=transform_function)
    print(transformer)
    transformer.test(grayscale=True)


if __name__ == '__main__':
    test_transformer()
