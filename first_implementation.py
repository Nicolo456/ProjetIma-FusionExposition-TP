import numpy as np
from display_func import show_image
from fs_func import open_image, save_image
from filters import apply_contrast_filter, apply_grayscale, apply_saturation_filter, apply_well_exposedness_filter


def calc_weight_map(contrast_weight_map, saturation_weight_map, well_exposedness_weight_map, contrast_power=0, saturation_power=0, well_exposedness_power=1, show=False, img=None):
    weight_map = (contrast_weight_map ** contrast_power) * (saturation_weight_map **
                                                            saturation_power) * (well_exposedness_weight_map ** well_exposedness_power)

    if show == True and img is not None:
        show_image(img, img1_title='Original Image', img2=weight_map,
                   img2_title='Final Weight Map', is_im2_grey=True)
    elif show == True and img is None:
        show_image(weight_map, img1_title='Final Weight Map', is_im1_grey=True)
    return weight_map


def get_weight_map(img, show=False):
    img_grayscale = apply_grayscale(img)

    # Compute the weight map, for each filter we want to normalize it between 0 and 1
    contrast_weight_map = apply_contrast_filter(img_grayscale, show=show)
    saturation_weight_map = apply_saturation_filter(img, show=show) / 120.3
    well_exposedness_weight_map = apply_well_exposedness_filter(
        img, show=show)
    weight_map = calc_weight_map(
        contrast_weight_map, saturation_weight_map, well_exposedness_weight_map, show=show, img=img)
    return weight_map


def get_weight_maps(imgs, show=False):
    weight_maps = []
    for img in imgs:
        weight_maps.append(get_weight_map(img, show=show))
    return weight_maps


def normalize_weight_maps(weight_maps, verbose=False):
    # Epsilon is a small value to avoid division by zero
    epsilon = 1e-10 * np.ones_like(weight_maps[0])

    px_sum_weight_maps = np.sum(weight_maps, axis=0) + epsilon

    # Si tous les poids sont nuls, alors le poids de chaque pixel sera réparti entre les images
    normalized_weight_maps = [
        np.divide(weight_map + epsilon/len(imgs), px_sum_weight_maps) for weight_map in weight_maps]

    if verbose:
        print("\nSum of Normalized weight maps:")
        print(np.sum(normalized_weight_maps, axis=0))
        print("Max of the sum:", np.max(np.sum(normalized_weight_maps, axis=0)))
        print("Min of the sum:", np.min(
            np.sum(normalized_weight_maps, axis=0)), "\n")

    return normalized_weight_maps


def fuse_images(imgs, normalized_weight_maps):
    # Copy the weight on every channel
    normalized_weight_maps_3d = [np.stack(
        [normalized_weight_map] * 3, axis=-1) for normalized_weight_map in normalized_weight_maps]

    # Fusionner les images en utilisant les poids normalisés
    fused_image = np.sum(
        [normalized_weight_map_3d * img for normalized_weight_map_3d, img in zip(normalized_weight_maps_3d, imgs)], axis=0)

    clipped_fused_image = np.clip(
        fused_image, 0, 255).astype(np.uint8)

    return clipped_fused_image


def naive_fusion(imgs, show=False):
    weight_maps = get_weight_maps(imgs, show=show)

    n_weight_maps = normalize_weight_maps(weight_maps, verbose=show)

    fused_image = fuse_images(imgs, n_weight_maps)

    return fused_image


if __name__ == "__main__":
    """ # Open an image with numpy, show it with matplotlib
    img_m = open_image("img/venise/MeanSat.jpg")
    img_o = open_image("img/venise/OverSat.jpg")
    img_u = open_image("img/venise/UnderSat.jpg")
    imgs = [img_m, img_o, img_u]

    fused_image = naive_fusion(imgs, show=False)

    # Chargement image du papier
    img_p = open_image("img/venise/Result.jpg")

    # Afficher l'image fusionnée
    show_image(img_p, img1_title='Image issue du papier',
               img2=fused_image, img2_title='Image issue de notre fusion naive')

    # Save the fused image
    save_image(fused_image, "img/venise/fused_image.jpg")

    # ============== Other example ===============
    imgs = []
    for i in range(4):
        imgs.append(open_image(f"img/chamber/iso{i + 1}.jpg"))

    fused_image = naive_fusion(imgs, show=False)

    # Chargement image du papier
    img_p = open_image("img/chamber/naive_paper_result.jpg")

    # Afficher l'image fusionnée
    show_image(img_p, img1_title='Image issue du papier',
               img2=fused_image, img2_title='Image issue de notre fusion naive')

    # Save the fused image
    save_image(fused_image, "img/chamber/fused_image.jpg") """

    imgs = []
    for k in range(259, 266):
        imgs.append(open_image(f"img/perso_dams/DSC08{k}.tiff"))

    fused_image = naive_fusion(imgs, show=False)

    # Save the fused image
    save_image(fused_image, "img/perso_dams/fused_image.jpg")
