import os
import sys
sys.path.append('..')

import cv2

import resources.utils as utils

# CREATE YOUR FUNCTION HERE
def transform_function(image):
    M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), 90, 1)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# CHANGE THE NAME OF THE FUNCTION HERE
def test_transformer():
    transformer = utils.CervAugmentor(name="Rotate", augment_func=transform_function)
    print(transformer)
    transformer.test() # Put grayscale=True if we expect a grayscale image transformed

if __name__ == '__main__':
    test_transformer()
