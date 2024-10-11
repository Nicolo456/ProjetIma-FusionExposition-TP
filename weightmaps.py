import numpy as np
from display_func import show_image, show_image_cv2, BGR2RGB, RGB2BGR
from fs_func import open_image, save_image
from filters import apply_contrast_filter, apply_grayscale, apply_saturation_filter, apply_well_exposedness_filter


def calc_wm(contrast_wm, saturation_wm, well_exposedness_wm, contrast_power=0, saturation_power=0, well_exposedness_power=1, show=False, img=None):
    wm = (contrast_wm ** contrast_power) * (saturation_wm **
                                                            saturation_power) * (well_exposedness_wm ** well_exposedness_power)

    if show == True and img is not None:
        show_image(img, img1_title='Original Image', img2=wm,
                   img2_title='Final Weight Map', is_im2_grey=True)
    elif show == True and img is None:
        show_image(wm, img1_title='Final Weight Map', is_im1_grey=True)
    return wm


def get_wm(img, show=False):
    img_grayscale = apply_grayscale(img)

    # Compute the weight map, for each filter we want to normalize it between 0 and 1
    contrast_wm = apply_contrast_filter(img_grayscale, show=show)
    saturation_wm = apply_saturation_filter(img, show=show) / 120.3
    well_exposedness_wm = apply_well_exposedness_filter(
        img, show=show)
    wm = calc_wm(
        contrast_wm, saturation_wm, well_exposedness_wm, show=show, img=img)
    return wm


def get_wms(imgs, show=False):
    wms = []
    for img in imgs:
        wms.append(get_wm(img, show=show))
    return wms


def normalize_wms(wms, verbose=False):
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


def fuse_images(imgs, normalized_wms):
    print("imgs", imgs)
    # Copy the weight on every channel
    normalized_wms_3d = [np.stack(
        [normalized_wm] * 3, axis=-1) for normalized_wm in normalized_wms]

    # Fusionner les images en utilisant les poids normalisés
    print("zip",list(zip(normalized_wms_3d, imgs)))    #tableau de (a,b) où a est une wm et b une liste d'images pour notre pb
    fused_image = np.sum(
        [normalized_wm_3d * img for normalized_wm_3d, img in zip(normalized_wms_3d, imgs)], axis=0)

    clipped_fused_image = np.clip(
        fused_image, 0, 255).astype(np.uint8)

    return clipped_fused_image


def naive_fusion(imgs, show=False):
    wms = get_wms(imgs, show=show)

    n_wms = normalize_wms(wms, verbose=show)

    fused_image = fuse_images(imgs, n_wms)

    return fused_image