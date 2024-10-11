import cv2
import matplotlib.pyplot as plt
from display_func import show_image, show_image_cv2, BGR2RGB, RGB2BGR


def pyramid_down(im, floors = 3, show = False):
    pyr = [im]
    if show == True:
        show_image_cv2(im)
    for i in range(floors-1):
        lower = cv2.pyrDown(pyr[i])
        pyr.append(lower)
        if show == True:
            show_image_cv2(lower)
    return pyr

def pyramid_up(im_down, floors = 3, show = False):
    pyr = [im_down]
    if show == True:
        show_image_cv2(im_down)
    for i in range(floors-1):
        higher = cv2.pyrUp(pyr[i])
        pyr.append(higher)
        if show == True:
            show_image_cv2(higher)
    return pyr

def laplacian_pyramid(im, floors = 3, show = False, rgb = True):
    pyr_down = pyramid_down(im, floors = floors)
    last_floor = pyr_down[floors-1]
    pyr = [last_floor]
    pyr_up = pyramid_up(last_floor, floors = floors)
    for i in range(floors-1):
        lapl_im = cv2.subtract(pyr_down[i],pyr_up[floors-i-1])
        pyr.append(lapl_im)
        if show:
            show_image_cv2(lapl_im)
    if rgb:
        for i in range(floors):
            pyr[i] = BGR2RGB(pyr[i])
            pyr_down[i] = BGR2RGB(pyr[i])
            pyr_up[i] = BGR2RGB(pyr[i])
    return pyr, pyr_down, pyr_up


"""
#Tests
img = cv2.imread('img/venise/MeanSat.jpg')
laplacian_pyramid(img,5,show=True)
pyr_down = pyramid_down(img,4,True)
N = len(pyr_down)
pyr_up = pyramid_up(pyr_down[N-1], 4, show = True)
"""


