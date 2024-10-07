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

def pyramid_up(im_down, floors = 0, show = False):
    if floors ==0:
        floors = len(pyr_down)
    pyr = [im_down]
    if show == True:
        show_image_cv2(im_down)
    for i in range(floors-1):
        higher = cv2.pyrUp(pyr[i])
        pyr.append(higher)
        if show == True:
            show_image_cv2(higher)
    return pyr



"""
#Tests
img = cv2.imread('img/venise/MeanSat.jpg')
pyr_down = pyramid_down(img,4,True)
N = len(pyr_down)
pyr_up = pyramid_up(pyr_down[N-1],show = True)
"""


