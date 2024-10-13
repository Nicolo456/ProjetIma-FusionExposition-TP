import numpy as np
from display_func import show_image, show_image_cv2, BGR2RGB, RGB2BGR, inspect_list_structure
from fs_func import open_image, save_image
from weightmaps import get_wms, normalize_wms, fuse_images
from pyramid import pyramid_down, reconstruct_from_lpyr, laplacian_pyramid

# TODO: Mettre des s partout au pyr parce que y en a plusieurs


def make_fused_summed_pyr(imgs, show=False, floors=3):
    """Wrapper used to generate the final pyramid like the paper

    @params: imgs: [image (np.array)] a list of image with different exposure
    @return: [[image (np.array)]] a pyramid of the final image"""

    wms = get_wms(imgs, show=show, forceFloat=True)

    imgs = [img.astype(np.float32) for img in imgs]

    imgs_pyrl = [laplacian_pyramid(
        img, floors=floors, show=show) for img in imgs]
    sorted_imgs_pyrl = sort_pyr(imgs_pyrl)

    wms_pyrg = [pyramid_down(wm) for wm in wms]
    sorted_n_wms_pyr = get_sorted_n_wms_pyrs(wms_pyrg)

    sorted_fused_pyrs = get_sorted_fused_pyrs(
        sorted_n_wms_pyr, sorted_imgs_pyrl)

    fused_summed_pyr = get_fused_summed_pyr(sorted_fused_pyrs)

    if show:
        for i in range(len(fused_summed_pyr)):
            show_image(fused_summed_pyr[i], img1_title='Fused image')

    return fused_summed_pyr


def sort_pyr(pyrs):
    """Sort pyramid by floor
    An utilitarie function used to make transformation by floor easier"""
    # Initialiser une liste vide pour stocker les groupes
    sorted_pyr = [[] for _ in range(len(pyrs[0]))]

    # Parcourir chaque sous-liste et regrouper les éléments par index
    for floor in pyrs:
        for i, img in enumerate(floor):
            sorted_pyr[i].append(img)
    return sorted_pyr


def get_sorted_n_wms_pyrs(wms_pyrs):
    """Return a list of lists representing the floor. Each floor group contain the floor of each weight_map pyramid (sorted_n_wms_pyrs)

    @param: wms_pyrs: [[image (np.array)]] a list of pyramid for each weight_map
    @return: [[image (np.array)]] a list of floor_group with normalised weight map of the floor (one by pyramid)"""
    sorted_wms_pyrs = sort_pyr(wms_pyrs)
    sorted_n_wms_pyrs = list(map(normalize_wms, sorted_wms_pyrs))
    return sorted_n_wms_pyrs


def get_sorted_fused_pyrs(sorted_n_wms_pyr, sorted_imgs_pyrl):
    """Return a list of all lists representing the floor. Each floor_group contain the floor of each fusion of weight_map and image pyramid

    @params: sorted_n_wms_pyrs: [[image (np.array)]] a list of floor_group with normalised weight map of the floor (one by pyramid)
    @params: sorted_imgs_pyrl: [[image (np.array)]] a list of floor_group with image pyramid of the floor (one by pyramid)
    @return: [[image (np.array)]] a list of floor_group with fused image of the floor (one by pyramid)"""

    sorted_fused_pyrs = list(map(fuse_images, sorted_imgs_pyrl,
                                 sorted_n_wms_pyr))  # On applique fuse image sur chaque pyramide

    return sorted_fused_pyrs


def get_fused_summed_pyr(sorted_fused_pyr):
    """On prend une liste de pyramide qu'on va mélanger avec une addition par étage
    @params: sorted_fused_pyr: [[image (np.array)]]"""

    fused_summed_pyr = []
    for pyr in sorted_fused_pyr:
        fused_summed_pyr.append(np.sum(pyr, axis=0))

    # fused_summed_pyr = list(map(lambda x: np.sum(x, axis=0), sorted_fused_pyr))
    return fused_summed_pyr


def get_exposition_fused_image(imgs, show=False):
    """Wrapper used to execute the paper algorithme

    @params: imgs: [image (np.array)] a list of image with different exposure
    @return: image (np.array) the final image"""

    # Execute the wrapper for the pyramid
    fused_summed_pyr = make_fused_summed_pyr(imgs, show=False)

    # fused_summed_pyr_bgr = [RGB2BGR(img) for img in fused_summed_pyr]
    final_image = reconstruct_from_lpyr(fused_summed_pyr)
    # final_image_rgb = BGR2RGB(final_image)
    if show:
        show_image(final_image, img1_title='Final image')
    return final_image


if __name__ == "__main__":
    # Open an image with numpy, show it with matplotlib
    img_m = open_image("img/venise/MeanSat.jpg")
    img_o = open_image("img/venise/OverSat.jpg")
    img_u = open_image("img/venise/UnderSat.jpg")
    imgs = [img_m, img_o, img_u]

    final_image = get_exposition_fused_image(imgs, show=False)
    show_image(final_image, img1_title='Final image')
