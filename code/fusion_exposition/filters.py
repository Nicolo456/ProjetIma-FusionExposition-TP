import numpy as np
from scipy import ndimage
from .display_func import show_image
from .assert_decorator import assert_normalized_image

"""
Ce module contient des fonctions permettant de:
- convertir une image en niveau de gris
- appliquer les filtres de contraste, saturation et bonne-exposition"""

@assert_normalized_image(negative=True)
def apply_grayscale(img):
    # Convert the image to grayscale
    img_grayscale = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    img_grayscale = img_grayscale.astype(np.float32)
    return img_grayscale


@assert_normalized_image()
def apply_contrast_filter(img_grayscale, show=False):
    '''Apply a Laplacian filter to the grayscale version of each image, and take the absolute value of the filter response [16].
    This yields a simple indicator C for contrast. It tends to assign a high weight to important elements such as edges and texture.
    A similar measure was used for multi-focus fusion for extended depth-of-field [19]'''

    # On floute pour enlever le bruit
    blurred = ndimage.gaussian_filter(img_grayscale, sigma=1)
    laplacian_filtered = ndimage.laplace(blurred)
    laplacian_filtered = np.absolute(laplacian_filtered)
    # if np.max(np.abs(laplacian_filtered)) > 1:
    #     print("laplacian_filtered normalized for image:", laplacian_filtered.shape,
    #           "max:", np.max(laplacian_filtered), "min:", np.min(laplacian_filtered))
    #     laplacian_filtered = (
    #         laplacian_filtered / np.max(np.abs(laplacian_filtered))).astype(np.float32)

    if show == True:
        # Display the original and the filtered image
        show_image(img_grayscale, img1_title='Original Image', is_im1_grey=True,
                   img2=laplacian_filtered, img2_title='Laplacian Filtered Image', is_im2_grey=True)
    return laplacian_filtered


@assert_normalized_image(negative=True)
def apply_saturation_filter(img_colored, show=False):
    '''As a photograph undergoes a longer exposure, the resulting colors become desaturated and eventually clipped. Saturated colors are desirable and make the image look vivid. We include a saturation measure S, which is computed as the standard deviation within the R, G and B channel, at each pixel.'''

    """ # Extract the R, G, B channels
    R = img_colored[:, :, 0]
    G = img_colored[:, :, 1]
    B = img_colored[:, :, 2]

    # Stack the R, G, B channels along the last axis
    rgb_stack = np.stack((R, G, B), axis=-1) """

    # Compute the standard deviation across the R, G, B channels for each pixel
    saturation_map = np.std(img_colored, axis=-1)
    # if np.max(np.abs(saturation_map)) > 1:
    #     print("saturation_map normalized for image:", saturation_map.shape,
    #           "max:", np.max(saturation_map), "min:", np.min(saturation_map))
    #     saturation_map = (
    #         saturation_map / np.max(np.abs(saturation_map))).astype(np.float32)

    if show == True:
        show_image(img_colored, img1_title='Original Image',
                   img2=saturation_map, img2_title='Saturation Map', is_im2_grey=True)
    return saturation_map


@assert_normalized_image(negative=True)
def apply_well_exposedness_filter_grayscale(img_channel, sigma=0.2):
    '''Looking at just the raw intensities within a channel, reveals how well a pixel is exposed.
    We want to keep intensities that are not near zero (underexposed) or one (overexposed). We weight each intensity i based on how close it is to 0.5 using a Gauss curve: exp(i−0.5)ˆ2/2σˆi , where σ equals 0.2 in our implementation. '''
    # Compute the Gaussian function
    we_map = np.exp(-((img_channel - 0.5) ** 2) / (2 * sigma ** 2))

    # if np.max(np.abs(we_map)) > 1:
    #     print("we_map normalized for image:", we_map.shape,
    #           "max:", np.max(we_map), "min:", np.min(we_map))
    #     we_map = (we_map / np.max(np.abs(we_map))).astype(np.float32)

    return we_map


@assert_normalized_image(negative=True)
def apply_well_exposedness_filter(img_colored, show=False, sigma=0.2):
    '''To account for multiple color channels, we apply the Gauss curve to each channel separately, and multiply the results, yielding the measure E'''

    # Extract the R, G, B channels
    R = img_colored[:, :, 0]
    G = img_colored[:, :, 1]
    B = img_colored[:, :, 2]

    # Apply the well-exposedness filter to each channel
    R_we_map = apply_well_exposedness_filter_grayscale(R, sigma=sigma)
    G_we_map = apply_well_exposedness_filter_grayscale(G, sigma=sigma)
    B_we_map = apply_well_exposedness_filter_grayscale(B, sigma=sigma)

    # Combine the results from each channel
    we_map = R_we_map * G_we_map * B_we_map
    if show == True:
        show_image(img_colored, img1_title='Original Image', img2=we_map,
                   img2_title='Well-Exposedness Map', is_im2_grey=True)
    return we_map
