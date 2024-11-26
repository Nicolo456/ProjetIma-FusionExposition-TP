import numpy as np
from .display_func import show_image, inspect_list_structure
from .weightmaps import get_wms, normalize_wms, fuse_and_sum_images
from .pyramid import pyramid_down, reconstruct_from_lpyr, laplacian_pyramid
from .normalization import normalise_vector_decorator
from .assert_decorator import assert_normalized_images, assert_normalized_pyrs, assert_image_size_divisible, get_max_floor_if_floor_Max


@assert_normalized_images()
def make_fused_summed_pyr(imgs, power_coef, show=False, floors=3):
    """Wrapper used to generate the final pyramid like the paper

    @params: imgs: [image (np.array)] a list of image with different exposure
    @params: power_coef: [float] a list of power coefficient for each weight map
        [contrast_power, saturation_power, well_exposedness_power]
    @params: show: bool, if True, show the weight map of the first image
    @return: [[image (np.array)]] a pyramid of the final image"""

    wms = get_wms(imgs, power_coef, show=show)

    imgs_pyrl = [laplacian_pyramid(
        img, floors=floors, show=show) for img in imgs]
    sorted_imgs_pyrsl = sort_pyr(imgs_pyrl)

    wms_pyrg = [pyramid_down(wm, floors=floors) for wm in wms]
    sorted_n_wms_pyrs = get_sorted_n_wms_pyrs(wms_pyrg)

    fused_summed_pyr = get_fused_summed_pyr(
        sorted_n_wms_pyrs, sorted_imgs_pyrsl)

    if show:
        for i in range(len(fused_summed_pyr)):
            show_image(fused_summed_pyr[i], img1_title='Fused image')

    return fused_summed_pyr


@assert_normalized_pyrs(negative=True)
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


@assert_normalized_pyrs(negative=True)
def get_sorted_n_wms_pyrs(wms_pyrs):
    """Return a list of lists representing the floor. Each floor group contain the floor of each weight_map pyramid (sorted_n_wms_pyrs)

    @param: wms_pyrs: [[image (np.array)]] a list of pyramid for each weight_map
    @return: [[image (np.array)]] a list of floor_group with normalised weight map of the floor (one by pyramid)"""
    sorted_wms_pyrs = sort_pyr(wms_pyrs)
    sorted_n_wms_pyrs = list(map(normalize_wms, sorted_wms_pyrs))
    return sorted_n_wms_pyrs


@assert_normalized_pyrs(nb_pyrs=2, negative=True)
def get_fused_summed_pyr(sorted_n_wms_pyr, sorted_imgs_pyrl):
    """Return a list of all lists representing the floor. Each floor_group contain the floor of each fusion of weight_map and image pyramid

    @params: sorted_n_wms_pyrs: [[image (np.array)]] a list of floor_group with normalised weight map of the floor (one by pyramid)
    @params: sorted_imgs_pyrl: [[image (np.array)]] a list of floor_group with image pyramid of the floor (one by pyramid)
    @return: [[image (np.array)]] a list of floor_group with fused image of the floor (one by pyramid)"""
    fused_summed_pyr = []

    assert len(sorted_n_wms_pyr) == len(
        sorted_imgs_pyrl), "Les pyramides ne sont pas de la même taille"
    for i_floor in range(len((sorted_n_wms_pyr))):
        # On applique fuse image sur chaque pyramide
        fused_summed_pyr.append(fuse_and_sum_images(sorted_imgs_pyrl[i_floor],
                                                    sorted_n_wms_pyr[i_floor]))
    return fused_summed_pyr


@get_max_floor_if_floor_Max
@assert_image_size_divisible
@normalise_vector_decorator(force_normalize_return=True)
@assert_normalized_images()
def get_exposition_fused_image(imgs, floors, power_coef, show=False, clip=True):
    """Wrapper used to execute the paper algorithme

    @params: imgs: [image (np.array)] a list of image with different exposure
    @params: power_coef: [float] a list of power coefficient for each weight map
        [contrast_power, saturation_power, well_exposedness_power]
    @return: image (np.array) the final image"""

    # Execute the wrapper for the pyramid
    fused_summed_pyr = make_fused_summed_pyr(
        imgs, power_coef, show=False, floors=floors)

    final_image = reconstruct_from_lpyr(fused_summed_pyr)

    if show:
        show_image(final_image, img1_title='Final image')
    return final_image


@assert_image_size_divisible
@normalise_vector_decorator(force_normalize_return=True)
@assert_normalized_images()
def get_exposition_fused_image_sep_channel(imgs, floors, power_coef, show=False, clip=True):
    """Wrapper used to execute the paper algorithme

    @params: imgs: [image (np.array)] a list of image with different exposure
    @params: power_coef: [float] a list of power coefficient for each weight map
        [contrast_power, saturation_power, well_exposedness_power]
    @return: image (np.array) the final image"""

    # Try to execute on each channel
    imgs_channels = [[img[:, :, i] for img in imgs] for i in range(3)]

    final_image_channels = []
    for imgs_channel in imgs_channels:
        fused_summed_pyr_channel = make_fused_summed_pyr(
            imgs_channel, power_coef, show=False, floors=floors)
        final_image_channel = reconstruct_from_lpyr(fused_summed_pyr_channel)

        final_image_channels.append(final_image_channel)

    final_image_sep_channel = np.stack(final_image_channels, axis=-1)

    if show:
        show_image(final_image_sep_channel,
                   img1_title='Final image sep channel')
    return final_image_sep_channel
