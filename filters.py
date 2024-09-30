import numpy as np
from scipy import ndimage
from display_func import show_image


def apply_grayscale(img):
    # Convert the image to grayscale
    img_grayscale = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img_grayscale


def apply_contrast_filter(img_grayscale, show=False):
    '''Apply a Laplacian filter to the grayscale version of each image, and take the absolute value of the filter response [16].
    This yields a simple indicator C for contrast. It tends to assign a high weight to important elements such as edges and texture.
    A similar measure was used for multi-focus fusion for extended depth-of-field [19]'''

    # Define the Laplacian kernel (3x3)
    laplacian_kernel = np.array([[0, -1, 0],
                                [-1, 4, -1],
                                [0, -1, 0]])

    # other possibel kernels:
    laplacian_kernel_bis = np.array([[-1, -1, -1],
                                     [-1, 8, -1],
                                     [-1, -1, -1]])

    # Apply the Laplacian filter using convolution
    laplacian_filtered = ndimage.convolve(img_grayscale, laplacian_kernel)

    # Take the absolute value to get rid of negative values
    laplacian_filtered = np.absolute(laplacian_filtered)

    if show == True:
        # Display the original and the filtered image
        show_image(img_grayscale, img1_title='Original Image', is_im1_grey=True,
                   img2=laplacian_filtered, img2_title='Laplacian Filtered Image', is_im2_grey=True)
    return laplacian_filtered


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

    if show == True:
        show_image(img_colored, img1_title='Original Image',
                   img2=saturation_map, img2_title='Saturation Map', is_im2_grey=True)
    return saturation_map


def apply_well_exposedness_filter_grayscale(img_channel, sigma=0.2):
    # Normalize the channel to range [0, 1]
    # Assuming channel values are in [0, 255]
    img_normalised = img_channel / 255.0
    # Compute the Gaussian function
    we_map = np.exp(-((img_normalised - 0.5) ** 2) / (2 * sigma ** 2))

    return we_map


def apply_well_exposedness_filter(img_colored, show=False, sigma=0.2):
    '''Looking at just the raw intensities within a channel, reveals how well a pixel is exposed.
    We want to keep intensities that are not near zero (underexposed) or one (overexposed). We weight each intensity i based on how close it is to 0.5 using a Gauss curve: exp(i−0.5)ˆ2/2σˆi , where σ equals 0.2 in our implementation. To account for multiple color channels, we apply the Gauss curve to each channel separately, and multiply the results, yielding the measure E'''

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
