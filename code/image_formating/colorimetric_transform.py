import numpy as np
import cv2

def compute_hist(img):
    """ATTENTION: travail avec des ints entre 0 et 255"""
    # Convert the image to grayscale (if it's not already)
    if (len(img.shape) > 2 and img.shape[2] == 3):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
    # Compute the histogram with 256 bins (for grayscale)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist


def match_histogram_to_cdf(image, reference_cdf):
    # Compute the histogram of the input image
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist_cdf = np.cumsum(hist) / hist.sum()

    # Create a mapping from the source to the reference
    mapping = np.interp(hist_cdf, reference_cdf, np.arange(256))

    # Apply the mapping to the input image
    matched_image = np.interp(
        image.flatten(), bins[:-1], mapping).reshape(image.shape)
    return matched_image


def correct_hist_grayscale(imgs, final_image):
    """ATTENTION: travail avec des ints entre 0 et 255
    travail avec des images grayscale (un channel)"""
    # Calculate histograms for all 4 images
    hists = [compute_hist(img) for img in imgs]

    # Calculate the mean histogram
    mean_hist = np.mean(hists, axis=0)
    # Normalize the mean histogram
    n_mean_hist = mean_hist / np.sum(mean_hist)
    # Compute the CDF of the mean histogram
    mean_cdf = np.cumsum(n_mean_hist)

    matched_image = match_histogram_to_cdf(final_image, mean_cdf)

    # Convert the result back to the original image format
    matched_image = np.uint8(matched_image)

    return matched_image


def correct_hist_rgb(imgs, final_image):
    """ATTENTION: travail avec des ints entre 0 et 255
    travail avec des images rgb"""
    # Calculate histograms for all 4 images
    matched_image = np.zeros_like(final_image)
    for i in range(3):
        imgs_1channel = [img[:, :, i] for img in imgs]
        matched_image[:, :, i] = correct_hist_grayscale(
            imgs_1channel, final_image[:, :, i])
    return matched_image


def correct_hist_hsv(imgs, final_image):
    """ATTENTION: travail avec des ints entre 0 et 255
    travail avec des images bgr mais on va passer en hsv"""
    # Calculate histograms for all 4 images
    hsv_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in imgs]
    # FINIR HISTOGRAMME POUR CHAQUE CHANNEL SV
    hsv_final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)

    for hsv_img in hsv_imgs:
        s_matched_image = correct_hist_grayscale(
            hsv_final_image[:, :, 1], hsv_img[:, :, 1])
        s_matched_image = hsv_final_image[:, :, 1]
        v_matched_image = correct_hist_grayscale(
            hsv_final_image[:, :, 2], hsv_img[:, :, 2])
        v_matched_image = hsv_final_image[:, :, 2]
    # Apply CLAHE to the Y channel

    # Merge the enhanced Y channel with the original U and V channels
    enhanced_hsv_final_img = cv2.merge(
        (hsv_final_image[:, :, 0], s_matched_image, v_matched_image))
    # Convert the image back to BGR color space
    enhanced_final_img = cv2.cvtColor(
        enhanced_hsv_final_img, cv2.COLOR_HSV2BGR)
    return enhanced_final_img


def apply_clahe_grey(img, clipLimit=0.5, tile_size=(64, 64)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tile_size)

    # Appliquer CLAHE à l'image
    enhanced_img = clahe.apply(img)
    return enhanced_img


def apply_clahe_hsv(img):
    # Convert the image to YUV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the YUV image into its channels
    h, s, v = cv2.split(hsv_img)
    # Apply CLAHE to the Y channel
    s_clahe = apply_clahe_grey(s, clipLimit=1.0)
    v_clahe = apply_clahe_grey(v)

    # Merge the enhanced Y channel with the original U and V channels
    enhanced_hsv_img = cv2.merge((h, s_clahe, v_clahe))
    # Convert the image back to BGR color space
    enhanced_img = cv2.cvtColor(enhanced_hsv_img, cv2.COLOR_HSV2BGR)
    return enhanced_img
