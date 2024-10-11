import numpy as np
from display_func import show_image, show_image_cv2, BGR2RGB, RGB2BGR
from fs_func import open_image, save_image
from filters import apply_contrast_filter, apply_grayscale, apply_saturation_filter, apply_well_exposedness_filter
from weightmaps import calc_wm, get_wm, get_wms, normalize_wms, fuse_images, naive_fusion
from pyramid import pyramid_down, pyramid_up, laplacian_pyramid

# Mettre des s partout au pyr parce que y en a plusieurs
def main(imgs, show=False):
    imgs_pyramids = [laplacian_pyramid(img) for img in imgs]

    wms = get_wms(imgs, show=False)
    
    imgs_pyrl  = [laplacian_pyramid(img) for img in imgs]
    sorted_imgs_pyrl = sort_pyr(imgs_pyrl)

    wms_pyrg = [pyramid_down(wm) for wm in wms]
    sorted_n_wms_pyr = get_sorted_n_wms_pyr(wms_pyrg)

    sorted_fused_pyr = get_sorted_fused_pyr(sorted_n_wms_pyr, sorted_imgs_pyrl)

    fused_summed_pyr = get_fused_summed_pyr(sorted_fused_pyr)

    if show:
        for i in range(len(fused_summed_pyr)):
            show_image(fused_summed_pyr[i], img1_title='Fused image')

    return fused_summed_pyr

def sort_pyr(pyrs):
    """Sort pyramid by floor
    
    pyrs = [[1, 2], [3, 4], [5, 6]]
    zip(*pyrs) -> [(1, 3, 5), (2, 4, 6)]
    map(list, zip(*pyrs)) -> [[1, 3, 5], [2, 4, 6]]"""
    sorted_pyr = list(map(list, zip(*pyrs)))
    return sorted_pyr


def get_sorted_n_wms_pyr(pyrs):                                         
    sorted_pyr = sort_pyr(pyrs)
    sorted_n_wms_pyr = list(map(normalize_wms, sorted_pyr))
    return sorted_n_wms_pyr

def get_sorted_fused_pyr(sorted_n_wms_pyr, sorted_imgs_pyrl):
    print(sorted_n_wms_pyr)
    fused_pyrs = list(map(fuse_images, sorted_imgs_pyrl, sorted_n_wms_pyr)) # On applique fuse image sur chaque pyramide
    sorted_fused_pyr = sort_pyr(fused_pyrs)
    return sorted_fused_pyr

def get_fused_summed_pyr(sorted_fused_pyr):
    """On prend une liste de pyramide qu'on va mélanger avec une addition par étage"""
    fused_summed_pyr = list(map(lambda x: np.sum(x, axis=0), sorted_fused_pyr))
    return fused_summed_pyr

def old_pyramid_fusion(imgs, show = False):
    assert 0 == 1 # Pour eviter de l'utiliser sans faire exprès

    imgs_pyramids = [laplacian_pyramid(img) for img in imgs]

    imgs_pyramids_group = list(map(list, zip(*imgs_pyramids)))

    wms = get_wms(imgs, show=show)

    weight_pyramids = [pyramid_down(wm) for wm in wms]

    weights_pyramids_group = list(map(list, zip(*weight_pyramids)))

    n_weight_pyramids = [normalize_wms(wms, verbose=show) for wms in weight_pyramids]

    n_weight_pyramids_3d = [[np.stack(
        [n_wm] * 3, axis=-1) for n_wm in n_wms] 
        for n_wms in n_weight_pyramids]
    
    n_weight_pyramids_3d_group = list(map(list, zip(*n_weight_pyramids_3d)))

    fused_weight_pyramid = [np.sum(
        [n_wm_3d * img for n_wm_3d, img in zip(n_wms_3d, imgs)], axis=0) 
        for n_wms_3d, imgs in zip(n_weight_pyramids_3d_group, imgs_pyramids_group)]

    
    clipped_fused_weight_pyramids = [np.clip(
        fused_weight, 0, 255).astype(np.uint8) for fused_weight in fused_weight_pyramid]
    
    return clipped_fused_weight_pyramids

if __name__ == "__main__":
     # Open an image with numpy, show it with matplotlib
    img_m = open_image("img/venise/MeanSat.jpg")
    img_o = open_image("img/venise/OverSat.jpg")
    img_u = open_image("img/venise/UnderSat.jpg")
    imgs = [img_m, img_o, img_u]

    fused_sum_pyr = main(imgs, show=True)

    """

    fused_image = naive_fusion(imgs, show=True)

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
    save_image(fused_image, "img/chamber/fused_image.jpg") 

    imgs = []
    for k in range(259, 266):
        imgs.append(open_image(f"img/perso_dams/DSC08{k}.tiff"))

    fused_image = naive_fusion(imgs, show=False)

    # Save the fused image
    save_image(fused_image, "img/perso_dams/fused_image.jpg")
    """