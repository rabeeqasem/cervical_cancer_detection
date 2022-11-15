import os
import sys
sys.path.append('..')

import cv2 as cv

import resources.utils as utils

# PUT FUNCTION HERE
def transform_function(image):
    return cv.resize(image, (224, 224))


def test_transformer():
    transformer = utils.CervTransformer(name="Resizing", transfunc=transform_function)
    print(transformer)
    transformer.test()


if __name__ == '__main__':
    test_transformer()
