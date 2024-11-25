import numpy as np
from display_func import show_image, show_image_cv2, BGR2RGB, RGB2BGR, inspect_list_structure
from fs_func import open_image, save_image
from filters import apply_contrast_filter, apply_grayscale, apply_saturation_filter, apply_well_exposedness_filter, apply_well_exposedness_filter_grayscale
from assert_decorator import assert_normalized_images, assert_normalized_image, is_img_greyscale


def calc_wm(contrast_wm, saturation_wm, well_exposedness_wm, contrast_power=1, saturation_power=1, well_exposedness_power=1, show=False, img=None):
    """Return a weight map for an image
    @params: img: image (np.array)
    @return: weight map (np.array)"""

    wm = (contrast_wm ** contrast_power) * (saturation_wm **
                                            saturation_power) * (well_exposedness_wm ** well_exposedness_power)

    if show == True and img is not None:
        show_image(img, img1_title='Original Image', img2=wm,
                   img2_title='Final Weight Map', is_im2_grey=True)
    elif show == True and img is None:
        show_image(wm, img1_title='Final Weight Map', is_im1_grey=True)
    return wm


@assert_normalized_image()
def get_wm(img, power_coef, show=False):
    """Return a weight map for an image
    @params: img: image (np.array)
    @return: weight map (np.array)"""

    if is_img_greyscale(img):
        # L'image est en noir et blanc
        contrast_wm = apply_contrast_filter(img, show=show)
        saturation_wm = np.ones_like(img)
        well_exposedness_wm = apply_well_exposedness_filter_grayscale(img)
        wm = calc_wm(
            contrast_wm, saturation_wm, well_exposedness_wm, contrast_power=power_coef[0], saturation_power=power_coef[1], well_exposedness_power=power_coef[2], show=show, img=img)
    else:
        # L'image est en couleur
        img_grayscale = apply_grayscale(img)

        # Compute the weight map, for each filter we want to normalize it between 0 and 1
        contrast_wm = apply_contrast_filter(img_grayscale, show=show)
        saturation_wm = apply_saturation_filter(img, show=show)
        well_exposedness_wm = apply_well_exposedness_filter(
            img, show=show)
        wm = calc_wm(
            contrast_wm, saturation_wm, well_exposedness_wm, contrast_power=power_coef[0], saturation_power=power_coef[1], well_exposedness_power=power_coef[2], show=show, img=img)
    return wm


@assert_normalized_images()
def get_wms(imgs, power_coef, show=False):
    """Return a list of weight maps for each image

    @params: imgs: [image (np.array)] a list of image with different exposure
    @params: show: bool, if True, show the weight map of the first image

    @return: [image (np.array)] a list of weight map"""

    wms = []
    for img in imgs:
        wms.append(get_wm(img, power_coef, show=show))
    return wms


@assert_normalized_images()
def normalize_wms(wms, verbose=False):
    """Normalize the weight map
    @params: wms: [image (np.array)] a list of weight map
    @params: verbose: bool, if True, print the sum of the weight map
    @return: [image (np.array)] a list of normalized weight map"""

    # Epsilon is a small value to avoid division by zero
    epsilon = 1e-10 * np.ones_like(wms[0])

    px_sum_wms = np.sum(wms, axis=0) + epsilon

    # Si tous les poids sont nuls, alors le poids de chaque pixel sera réparti entre les images
    normalized_wms = [
        np.divide(wm + epsilon/len(wms), px_sum_wms) for wm in wms]

    if verbose:
        print("\nSum of Normalized weight maps:")
        print(np.sum(normalized_wms, axis=0))
        print("Max of the sum:", np.max(np.sum(normalized_wms, axis=0)))
        print("Min of the sum:", np.min(
            np.sum(normalized_wms, axis=0)), "\n")

    return normalized_wms


@assert_normalized_images(2, negative=True)
def fuse_and_sum_images(imgs, normalized_wms):
    """Fusionner et sommed les images en utilisant les poids normalisés
    @params: imgs: [image (np.array)] a list of image with different exposure
    @params: normalized_wms: [image (np.array)] a list of normalized weight map
    @return: image (np.array) the fused image"""

    if is_img_greyscale(imgs[0]):
        fused_image = np.sum(
            [normalized_wm * img for normalized_wm, img in zip(normalized_wms, imgs)], axis=0)

    else:
        # Copy the weight on every channel
        normalized_wms_3d = [np.stack(
            [normalized_wm] * 3, axis=-1) for normalized_wm in normalized_wms]

        # Fusionner les images en utilisant les poids normalisés
        # print("zip",list(zip(normalized_wms_3d, imgs)))
        # tableau de (a,b) où a est une wm et b une liste d'images pour notre pb

        fused_image = np.sum(
            [normalized_wm_3d * img for normalized_wm_3d, img in zip(normalized_wms_3d, imgs)], axis=0)

    return fused_image


@assert_normalized_images()
def naive_fusion(imgs, power_coef, show=False):
    """Naive fusion of images
    @params: imgs: [image (np.array)] a list of image with different exposure
    @params: show: bool, if True, show the weight map of the first image
    @return: image (np.array) the fused image"""

    wms = get_wms(imgs, power_coef, show=show)

    n_wms = normalize_wms(wms, verbose=show)

    fused_image = fuse_and_sum_images(imgs, n_wms, forceInt=True)

    return fused_image


if __name__ == "__main__":
    # Open an image with numpy, show it with matplotlib
    img_m = open_image("img/trans_dams/med_aligned.tiff")
    img_o = open_image("img/trans_dams/over_aligned.tiff")
    img_u = open_image("img/trans_dams/under_aligned.tiff")

    img_m = open_image("img/venise/MeanSat.jpg")
    img_o = open_image("img/venise/OverSat.jpg")
    img_u = open_image("img/venise/UnderSat.jpg")
    imgs = [img_m, img_o, img_u]

    fused_image = naive_fusion(imgs, power_coef=[1, 1, 1], show=False)
    show_image(fused_image)
