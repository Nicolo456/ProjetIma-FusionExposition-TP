from fusion_exposition.fs_func import open_image, save_image
from image_formating.colorimetric_transform import add_saturation, correct_hist_hsv, correct_hist_rgb, apply_clahe_hsv
from datetime import datetime
from fusion_exposition.implementation import get_exposition_fused_image
from fusion_exposition.display_func import show_image
from image_formating.image_transform import unsharp_mask
from func.arg_parser import init_arg_parser
from func.date_handler import get_latest_file, time_counter
from os import listdir
from image_formating.downsampling import upsample_image
from os.path import join
import os
from PIL import Image # type: ignore
# from image_formating.compression import compress_image
from fusion_exposition.weightmaps import naive_fusion

LOGS_FOLDER = join("hidden_imgs", "logs")
LOGS_SAVE_NAME = "final_image"
DATE_FORMAT = "%Y-%m-%d:%H-%M-%S"
CURRENT_TIME = datetime.now().strftime(DATE_FORMAT)


def get_compressed_image(imgs, power_coef, args, save=False, is_naive_fusion=False, show=False):
    # ==================================== Fusion or fetch last image ============================
    i, j, k = power_coef
    if not args.reuse:
        @time_counter
        def timed_fusion():
            if is_naive_fusion:
                return naive_fusion(imgs, power_coef, show=show)
            else:
                return get_exposition_fused_image(
                    imgs, "Max", power_coef, show=show)

        final_image = timed_fusion()

        #compressed_image = compress_image(final_image, division_factor=3)
        if save:
            os.makedirs(LOGS_FOLDER, exist_ok=True)
            final_image_PIL = Image.fromarray(final_image)
            # fmt: off
            if is_naive_fusion:
                final_image_PIL.save(f"{join(LOGS_FOLDER, LOGS_SAVE_NAME)}_{CURRENT_TIME}_naive_c{i**2}s{j**2}we{k**2}.jpg", "JPEG")
                print(f"Image saved: {join(LOGS_FOLDER, LOGS_SAVE_NAME)}_{CURRENT_TIME}_naive_c{i**2}s{j**2}we{k**2}.jpg")
            else:
                final_image_PIL.save(f"{join(LOGS_FOLDER, LOGS_SAVE_NAME)}_{CURRENT_TIME}_naive_c{i**2}s{j**2}we{k**2}.jpg", "JPEG")
                print(f"Image saved: {join(LOGS_FOLDER, LOGS_SAVE_NAME)}_sharp_{CURRENT_TIME}.tiff")
            # fmt: on
        return final_image
        # Save the image into the logs folder
        # save_image(compressed_image, f"{join(LOGS_FOLDER, LOGS_SAVE_NAME)}_{CURRENT_TIME}_c{i**2}s{j**2}we{k**2}.jpg")
    else:
        # On réouvre la dernière image enregistrée
        final_image_name = get_latest_file(
            listdir(LOGS_FOLDER), DATE_FORMAT)
        final_image = open_image(
            f"{join(LOGS_FOLDER, final_image_name)}")
        return final_image


def init_img_dam():
    img_m = open_image("imgs/trans_dams/med_aligned.tiff")
    img_o = open_image("imgs/trans_dams/over_aligned.tiff")
    img_u = open_image("imgs/trans_dams/under_aligned.tiff")
    img_mo = open_image("imgs/trans_dams/med_over_aligned.tiff")
    imgs = [img_m, img_o, img_u, img_mo]
    return imgs, img_m


def init_img_venise(nb_floors=7):
    img_m = upsample_image("imgs/venise/MeanSat.jpg",
                           two_divisibily_factor=nb_floors)
    img_o = upsample_image("imgs/venise/OverSat.jpg",
                           two_divisibily_factor=nb_floors)
    img_u = upsample_image("imgs/venise/UnderSat.jpg",
                           two_divisibily_factor=nb_floors)
    imgs = [img_m, img_o, img_u]
    return imgs, img_m


def init_img_foyer(nb_floors=8):
    PATH_DIR = "imgs/foyer/"
    imgs_path = listdir(PATH_DIR)

    imgs = [upsample_image(f"{PATH_DIR}{img_i_path}",
                           two_divisibily_factor=nb_floors) for img_i_path in imgs_path if img_i_path != ".DS_Store"]
    img_m = imgs[3]
    return imgs, img_m

def view_exposition_modified_image(final_image, imgs, args, sharped=False, save=False):
    if save:
        save_image(
            final_image, f"{join(LOGS_FOLDER, LOGS_SAVE_NAME)}_{CURRENT_TIME}.tiff")
        print(f"Image saved: {join(LOGS_FOLDER, LOGS_SAVE_NAME)}_{CURRENT_TIME}.tiff")
    if sharped:
        previous_final_image = final_image
        final_image = unsharp_mask(
            final_image, sigma=0.8, strength=1.5, kernel_size=(5, 5))
        show_image(final_image, img1_title='Final image sharp',
                   img2=previous_final_image, img2_title='Final image')
        if save:
            save_image(
            final_image, f"{join(LOGS_FOLDER, LOGS_SAVE_NAME)}_sharp_{CURRENT_TIME}.tiff")
            print(f"Image saved: {join(LOGS_FOLDER, LOGS_SAVE_NAME)}_sharp_{CURRENT_TIME}.tiff")
    # ==================================== If flag apply transform ============================
    if args.exposition:
        corrected_hist_final_image = correct_hist_hsv(
            imgs, final_image)
        show_image(corrected_hist_final_image,
                   img1_title='Corrected hist hsv Final image')
        if save:
            save_image(
            corrected_hist_final_image, f"{join(LOGS_FOLDER, LOGS_SAVE_NAME)}_sharp_hsv_{CURRENT_TIME}.tiff")
            print(f"Image saved: {join(LOGS_FOLDER, LOGS_SAVE_NAME)}_sharp_hsv_{CURRENT_TIME}.tiff")

        # ============================== CLAHE ==============================
        clahe_final_image = apply_clahe_hsv(final_image)
        show_image(clahe_final_image,
                   img1_title='CLAHE Final image')
        if save:
            save_image(
            clahe_final_image, f"{join(LOGS_FOLDER, LOGS_SAVE_NAME)}_CLAHE_{CURRENT_TIME}.tiff")
            print(f"Image saved: {join(LOGS_FOLDER, LOGS_SAVE_NAME)}_CLAHE_{CURRENT_TIME}.tiff")



if __name__ == "__main__":
    args = init_arg_parser()
    # ==================================== Open images ==========================================
    # ---------------------------------- Image à la montagne avec Damien ----------------------------
    # imgs, img_m = init_img_dam()

    # ----------------------------- Image Venise du papier de recherche ----------------------------
    # imgs, img_m = init_img_venise()

    # ----------------------------- Image dans le foyer avec Billard ----------------------------
    imgs, img_m = init_img_foyer()

    # ---------------------------------------------------------------------------------------------

    # ------------------------------------- MODIFIER ICI -------------------------------------------
    """Description des arguments à modifier:
    imgs: listes des images utilisé dans la fusion
    power_coef: liste des coefficients de puissance pour les différents paramètres
        ↳ dans cette order [contrast_power, saturation_power, well_exposedness_power]
    is_naive_fusion: booléen, si True, utilise la fusion naïve, sinon utilise la fusion avec les poids
     """
    result = get_compressed_image(
        imgs, [1, 1, 1], args, is_naive_fusion=True, save=True)
    show_image(result, img1_title='Resultat final')
    # ----------------------------------------------------------------------------------------------

    # for i in range(1, 3):
    #     for j in range(1, 3):
    #         for k in range(1, 3):
    #             # [contrast_power, saturation_power, well_exposedness_power]
    #             # When the power augments, it will more effect the weight map, if the coefficient is 0, it will not effect the weight map.
    #             power_coef = [i**3, j**3, k**3]
    #             final_image = get_compressed_image(
    #                 imgs, power_coef, args, save=True, show=False)

    # ==================================== Show Result ========================================
    # show_image(final_image, img1_title='Final image')