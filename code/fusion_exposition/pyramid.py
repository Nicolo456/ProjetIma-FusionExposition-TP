import cv2
import matplotlib.pyplot as plt
from .display_func import show_image, show_image_cv2, BGR2RGB, RGB2BGR, inspect_list_structure
from .assert_decorator import assert_normalized_pyr

"""
Ce module contient des fonctions permettant de:
- construire une pyramide d'images (laplacienne ou gaussienne)
- reconstruire une image à partir d'une pyramide"""


def pyramid_gaussienne(im, floors=3, show=False):
    """
        Construct an image pyramid by downsampling.

        Args:
            im (numpy.ndarray): Input image to start downsampling from.
            floors (int): Number of pyramid levels. Default is 3.
            show (bool): If True, display intermediate results. Default is False.

        Returns:
            list: Pyramid levels, with each level being a downsampled version of the previous one.
    """
    pyr = [im]
    if show == True:
        show_image_cv2(im)
    for i in range(floors-1):

        lower = cv2.pyrDown(pyr[i])
        pyr.append(lower)
        if show == True:
            show_image_cv2(lower)
    return pyr


def pyramid_up(im_down, floors=3, show=False):
    """
        Construct an image pyramid by upsampling.

        Args:
            im_down (numpy.ndarray): Input image to start upsampling from.
            floors (int): Number of pyramid levels. Default is 3.
            show (bool): If True, display intermediate results. Default is False.

        Returns:
            list: Pyramid levels, with each level being an upsampled version of the previous one.
    """
    pyr = [im_down]
    if show == True:
        show_image_cv2(im_down)
    for i in range(floors-1):
        higher = cv2.pyrUp(pyr[i])
        pyr.append(higher)
        if show == True:
            show_image_cv2(higher)
    return pyr


@assert_normalized_pyr(negative=True)
def reconstruct_from_lpyr(pyr, show=False):
    """
        Reconstruct an image from a Laplacian pyramid. (First level is the highest resolution)

        Args:
            pyr (list): Laplacian pyramid levels.
            show (bool): If True, display intermediate results. Default is False.

        Returns:
                numpy.ndarray: Reconstructed image.
    """
    reconstructed = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        reconstructed = cv2.pyrUp(reconstructed)
        reconstructed = cv2.add(reconstructed, pyr[i])
        if show:
            show_image_cv2(reconstructed)
    return reconstructed


def laplacian_pyramid(im, floors=3, show=False):
    """
    Compute the Laplacian pyramid of an image.

    Args:
        im (numpy.ndarray): Input image.
        floors (int): Number of pyramid levels. Default is 3.
        show (bool): If True, display intermediate results. Default is False.

    Returns:
        list: Laplacian pyramid levels.
    """
    pyr_down = pyramid_gaussienne(im, floors=floors)

    last_floor = pyr_down[floors-1]
    pyr = []
    pyr_up = pyramid_up(last_floor, floors=floors)

    for i in range(floors-1):
        lapl_im = cv2.subtract(pyr_down[i], cv2.resize(
            pyr_up[floors - i - 1], (pyr_down[i].shape[1], pyr_down[i].shape[0])))
        pyr.append(lapl_im)
        if show:
            show_image_cv2(lapl_im)
    pyr.append(last_floor)
    return pyr


if __name__ == "__main__":
    img = cv2.imread('C:/Damien/Cours/Telecom Paris/Cours_2A/IMA/4IM01/depot_Nico/code/imgs/foyer/3med.tiff')
    laplacian_pyramid(img, 5, show=True)
    pyr_down = pyramid_gaussienne(img, 4, True)
    N = len(pyr_down)
    pyr_up = pyramid_up(pyr_down[N-1], 4, show=True)
