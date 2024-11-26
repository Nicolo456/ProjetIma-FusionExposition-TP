# To import from a file in the parent directory
from ..fusion_exposition.assert_decorator import assert_normalized_image, assert_normalized_images
import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Verify that work with a greyscale image and a colored image
# Test the assert_normalized_image decorator


@assert_normalized_image()
def function2test(image):
    return image

# Test the assert_normalized_images decorator


@assert_normalized_images()
def function_vector2test(images):
    return images


image_grey_normalized = np.array([[0.5, 0.7], [0.2, 0.9]], dtype=np.float32)
image_grey_unnormalized = np.array([[1.5, 0.7], [0.2, 0.9]], dtype=np.float32)
image_color_normalized = np.array(
    [[[0.5, 0.7, 0.2], [0.2, 0.9, 0.2]], [[0.3, 0.6, 0.2], [0.1, 0.8, 0.2]]], dtype=np.float32)
image_color_unnormalized = np.array(
    [[[1.5, 0.7, 0.2], [0.2, 0.9, 0.2]], [[0.3, 0.6, 0.2], [0.1, 0.8, 0.2]]], dtype=np.float32)

images_grey_normalized = [image_grey_normalized, image_grey_normalized]
images_grey_unnormalized = [image_grey_unnormalized, image_grey_unnormalized]
images_color_normalized = [image_color_normalized, image_color_normalized]
images_color_unnormalized = [
    image_color_unnormalized, image_color_unnormalized]


def test_assert_normalized_image():
    try:
        function2test(image_grey_normalized)
    except AssertionError as error:
        assert False, "AssertionError raised for normalized greyscale image"

    try:
        function2test(image_grey_unnormalized)
    except AssertionError as error:
        pass
    else:
        assert False, "AssertionError not raised for unnormalized greyscale image"

    try:
        function2test(image_color_normalized)
    except AssertionError as error:
        assert False, "AssertionError raised for normalized colored image"

    try:
        function2test(image_color_unnormalized)
    except AssertionError as error:
        pass
    else:
        assert False, "AssertionError not raised for unnormalized colored image"


def test_assert_normalized_images():
    try:
        function_vector2test(images_grey_normalized)
    except AssertionError as error:
        assert False, "AssertionError raised for normalized greyscale images"

    try:
        function_vector2test(images_grey_unnormalized)
    except AssertionError as error:
        pass
    else:
        assert False, "AssertionError not raised for unnormalized greyscale images"

    try:
        function_vector2test(images_color_normalized)
    except AssertionError as error:
        assert False, "AssertionError raised for normalized colored images"

    try:
        function_vector2test(images_color_unnormalized)
    except AssertionError as error:
        pass
    else:
        assert False, "AssertionError not raised for unnormalized colored images"
