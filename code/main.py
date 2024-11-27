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

LOGS_FOLDER = "hidden_imgs/logs/"
LOGS_SAVE_NAME = "final_image"
DATE_FORMAT = "%Y-%m-%d:%H-%M-%S"
CURRENT_TIME = datetime.now().strftime(DATE_FORMAT)

if __name__ == "__main__":
    args = init_arg_parser()

    # ==================================== Open images ==========================================
    # Open an image with numpy, show it with matplotlib
    # img_m = open_image("imgs/trans_dams/med_aligned.tiff")
    # img_o = open_image("imgs/trans_dams/over_aligned.tiff")
    # img_u = open_image("imgs/trans_dams/under_aligned.tiff")
    # img_mo = open_image("imgs/trans_dams/med_over_aligned.tiff")
    # imgs = [img_m, img_o, img_u, img_mo]

    nb_floors = 7
    img_m = upsample_image("hidden_imgs/venise/MeanSat.jpg",
                           two_divisibily_factor=nb_floors)
    img_o = upsample_image("hidden_imgs/venise/OverSat.jpg",
                           two_divisibily_factor=nb_floors)
    img_u = upsample_image("hidden_imgs/venise/UnderSat.jpg",
                           two_divisibily_factor=nb_floors)
    imgs = [img_m, img_o, img_u]

    # PATH_DIR = "imgs/foyer/"
    # imgs_path = listdir(PATH_DIR)

    # imgs = [upsample_image(f"{PATH_DIR}{img_i_path}",
    #                        two_divisibily_factor=8) for img_i_path in imgs_path if img_i_path != ".DS_Store"]
    # img_m = imgs[3]

    # ==================================== Fusion or fetch last image ============================
    if not args.reuse:
        # [contrast_power, saturation_power, well_exposedness_power]
        # When the power augments, it will more effect the weight map, if the coefficient is 0, it will not effect the weight map.
        power_coef = [1, 10, 1]

        @time_counter
        def timed_fusion():
            return get_exposition_fused_image(
                imgs, "Max", power_coef, show=True)

        final_image = timed_fusion()

        # Save the image into the logs folder
        save_image(
            final_image, f"{LOGS_FOLDER}{LOGS_SAVE_NAME}_{CURRENT_TIME}.tiff")
    else:
        # On réouvre la dernière image enregistrée
        final_image_name = get_latest_file(listdir(LOGS_FOLDER), DATE_FORMAT)
        final_image = open_image(f"{LOGS_FOLDER}{final_image_name}")

    # ==================================== Show Result ========================================
    show_image(final_image, img1_title='Final image')
    previous_final_image = final_image
    final_image = unsharp_mask(
        final_image, sigma=0.8, strength=1.5, kernel_size=(5, 5))
    show_image(final_image, img1_title='Final image sharp',
               img2=previous_final_image, img2_title='Final image')
    save_image(
        final_image, f"{LOGS_FOLDER}{LOGS_SAVE_NAME}_sharp_{CURRENT_TIME}.tiff")

    # ==================================== If flag apply transform ============================
    if args.exposition:
        # ====================== Histogram equalization =====================
        corrected_hist_final_image = correct_hist_rgb(imgs, final_image)
        show_image(corrected_hist_final_image,
                   img1_title='Corrected hist Final image')

        corrected_hist_final_image = correct_hist_rgb([img_m], final_image)
        show_image(corrected_hist_final_image,
                   img1_title='Corrected hist for med Final image')

        corrected_hist_final_image = correct_hist_hsv(imgs, final_image)
        show_image(corrected_hist_final_image,
                   img1_title='Corrected hist hsv Final image')
        save_image(
            corrected_hist_final_image, f"{LOGS_FOLDER}{LOGS_SAVE_NAME}_sharp_hsv_{CURRENT_TIME}.tiff")

        corrected_hist_final_image = correct_hist_hsv([img_m], final_image)
        show_image(corrected_hist_final_image,
                   img1_title='Corrected hist for med hsv Final image')
        save_image(
            corrected_hist_final_image, f"{LOGS_FOLDER}{LOGS_SAVE_NAME}_sharp_hsv_med_{CURRENT_TIME}.tiff")

        # ============================== CLAHE ==============================
        clahe_final_image = apply_clahe_hsv(final_image)
        show_image(clahe_final_image, img1_title='CLAHE Final image')
        save_image(
            clahe_final_image, f"{LOGS_FOLDER}{LOGS_SAVE_NAME}_CLAHE_{CURRENT_TIME}.tiff")

        # ============================== SATURATION ==============================
        sat_final_image = add_saturation(final_image)
        show_image(sat_final_image, img1_title='SATURATION Final image')
        save_image(
            sat_final_image, f"{LOGS_FOLDER}{LOGS_SAVE_NAME}_Saturation_{CURRENT_TIME}.tiff")
