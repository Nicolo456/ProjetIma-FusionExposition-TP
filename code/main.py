from fusion_exposition.fs_func import open_image, save_image
from image_formating.colorimetric_transform import correct_hist_hsv, correct_hist_rgb, apply_clahe_hsv
from datetime import datetime
from fusion_exposition.implementation import get_exposition_fused_image
from fusion_exposition.display_func import show_image
from image_formating.colorimetric_transform import correct_hist_hsv, correct_hist_rgb, apply_clahe_hsv
from image_formating.image_transform import unsharp_mask
from func.arg_parser import init_arg_parser, is_reuse_arg_present
from func.date_handler import get_latest_file
from os import listdir

LOGS_FOLDER = "img/logs/"
LOGS_SAVE_NAME = "final_image"
DATE_FORMAT = "%Y-%m-%d:%H-%M-%S"

if __name__ == "__main__":
    args = init_arg_parser()

    if not is_reuse_arg_present(args):
        # Open an image with numpy, show it with matplotlib
        img_m = open_image("img/trans_dams/med_aligned.tiff")
        img_o = open_image("img/trans_dams/over_aligned.tiff")
        img_u = open_image("img/trans_dams/under_aligned.tiff")
        img_mo = open_image("img/trans_dams/med_over_aligned.tiff")
        imgs = [img_m, img_o, img_u, img_mo]

        # img_m = open_image("img/venise/MeanSat.jpg")
        # img_o = open_image("img/venise/OverSat.jpg")
        # img_u = open_image("img/venise/UnderSat.jpg")
        # imgs = [img_m, img_o, img_u]

        # [contrast_power, saturation_power, well_exposedness_power]
        # When the power augments, it will more effect the weight map, if the coefficient is 0, it will not effect the weight map.
        power_coef = [1, 10, 1]

        final_image = get_exposition_fused_image(
            imgs, "Max", power_coef, show=False)
        show_image(final_image, img1_title='Final image')
    
    else:
        # Open an image with numpy, show it with matplotlib
        img_m = open_image("img/trans_dams/med_aligned.tiff")
        img_o = open_image("img/trans_dams/over_aligned.tiff")
        img_u = open_image("img/trans_dams/under_aligned.tiff")
        img_mo = open_image("img/trans_dams/med_over_aligned.tiff")
        imgs = [img_m, img_o, img_u, img_mo]

        # img_m = open_image("img/venise/MeanSat.jpg")
        # img_o = open_image("img/venise/OverSat.jpg")
        # img_u = open_image("img/venise/UnderSat.jpg")
        # imgs = [img_m, img_o, img_u]
        
        # On réouvre la dernière image enregistrée
        final_image_name= get_latest_file(listdir(LOGS_FOLDER), DATE_FORMAT)
        final_image = open_image(f"{LOGS_FOLDER}{final_image_name}")

    previous_final_image = final_image
    final_image = unsharp_mask(
        final_image, sigma=0.8, strength=1.5, kernel_size=(5, 5))
    show_image(final_image, img1_title='Final image sharp',
                img2=previous_final_image, img2_title='Final image')

    corrected_hist_final_image = correct_hist_rgb(imgs, final_image)
    show_image(corrected_hist_final_image,
                img1_title='Corrected hist Final image')

    corrected_hist_final_image = correct_hist_rgb([img_m], final_image)
    show_image(corrected_hist_final_image,
                img1_title='Corrected hist for med Final image')

    corrected_hist_final_image = correct_hist_hsv(imgs, final_image)
    show_image(corrected_hist_final_image,
                img1_title='Corrected hist hsv Final image')

    corrected_hist_final_image = correct_hist_hsv([img_m], final_image)
    show_image(corrected_hist_final_image,
                img1_title='Corrected hist for med hsv Final image')

    clahe_final_image = apply_clahe_hsv(final_image)
    show_image(clahe_final_image, img1_title='CLAHE Final image')

    # Save the image into the logs folder
    current_time = datetime.now().strftime(DATE_FORMAT)
    save_image(
        final_image, f"{LOGS_FOLDER}{LOGS_SAVE_NAME}_{current_time}.tiff")