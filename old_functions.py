import numpy as np
from weightmaps import get_wms, normalize_wms
from pyramid import pyramid_down, laplacian_pyramid


def old_pyramid_fusion(imgs, show=False):
    assert 0 == 1  # Pour eviter de l'utiliser sans faire exprès

    imgs_pyramids = [laplacian_pyramid(img) for img in imgs]

    imgs_pyramids_group = list(map(list, zip(*imgs_pyramids)))

    wms = get_wms(imgs, show=show)

    weight_pyramids = [pyramid_down(wm) for wm in wms]

    weights_pyramids_group = list(map(list, zip(*weight_pyramids)))

    n_weight_pyramids = [normalize_wms(wms, verbose=show)
                         for wms in weight_pyramids]

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
