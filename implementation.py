import numpy as np
from display_func import show_image, show_image_cv2, BGR2RGB, RGB2BGR
from fs_func import open_image, save_image
from filters import apply_contrast_filter, apply_grayscale, apply_saturation_filter, apply_well_exposedness_filter
from weightmaps import calc_weight_map, get_weight_map, get_weight_maps, normalize_weight_maps, fuse_images, naive_fusion
from pyramid import pyramid_down, pyramid_up, laplacian_pyramid


def pyramid_fusion(imgs, show = False):

    imgs_pyramids = [laplacian_pyramid(img) for img in imgs]

    imgs_pyramids_group = list(map(list, zip(*imgs_pyramids)))

    weight_maps = get_weight_maps(imgs, show=show)

    weight_pyramids = [pyramid_down(weight_map) for weight_map in weight_maps]

    n_weight_pyramids = [normalize_weight_maps(weight_maps, verbose=show) for weight_maps in weight_pyramids]

    n_weight_pyramids_3d = [[np.stack(
        [n_weight_map] * 3, axis=-1) for n_weight_map in n_weight_maps] 
        for n_weight_maps in n_weight_pyramids]
    
    n_weight_pyramids_3d_group = list(map(list, zip(*n_weight_pyramids_3d)))

    fused_weight_pyramid = [np.sum(
        [n_weight_map_3d * img for n_weight_map_3d, img in zip(n_weight_maps_3d, imgs)], axis=0) 
        for n_weight_maps_3d, imgs in zip(n_weight_pyramids_3d_group, imgs_pyramids_group)]

    
    clipped_fused_weight_pyramids = [np.clip(
        fused_weight, 0, 255).astype(np.uint8) for fused_weight in fused_weight_pyramid]
    
    return clipped_fused_weight_pyramids


img_m = open_image("img/venise/MeanSat.jpg")
img_o = open_image("img/venise/OverSat.jpg")
img_u = open_image("img/venise/UnderSat.jpg")
imgs = [img_m, img_o, img_u]
result = pyramid_fusion(imgs)
for img in result:
    show_image(img)
    


"""
if __name__ == "__main__":
     # Open an image with numpy, show it with matplotlib
    img_m = open_image("img/venise/MeanSat.jpg")
    img_o = open_image("img/venise/OverSat.jpg")
    img_u = open_image("img/venise/UnderSat.jpg")
    imgs = [img_m, img_o, img_u]

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