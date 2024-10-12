import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_image(img1, img1_title='Original Image', is_im1_grey=False, img2=None, img2_title='Filtered Image', is_im2_grey=False):
    if img2 is not None:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10, 5), sharex=True, sharey=True)
        ax1.axis('off')
        ax2.axis('off')

        # Display the original and the filtered image
        ax1.set_title(img1_title)
        if is_im1_grey:
            ax1.imshow(
                img1, cmap='gray')
        else:
            ax1.imshow(img1)

        ax2.set_title(img2_title)
        if is_im2_grey:
            ax2.imshow(img2, cmap='gray')
        else:
            ax2.imshow(img2)

    else:
        plt.title(img1_title)
        plt.axis('off')
        if is_im1_grey:
            plt.imshow(img1, cmap='gray')
        else:
            plt.imshow(img1)

    plt.show()


def show_image_cv2(im, title = "Image"):          #cv2 travaille au format BGR et pas RGB 
    cv2.imshow(title, im)
    cv2.waitKey(0)              #0 attend l'appui d'une touche pour continuer, on peut mettre un temps en ms


def BGR2RGB(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image_rgb

def RGB2BGR(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return image_rgb

def inspect_list_structure(lst, level=1):
    """Inspect the structure of a list, identifying arrays, sublists, and other types."""
    """List of length 4
        Element 0:
            Numpy Array with shape (64, 64, 3), dtype float64
        Element 1:
            List of length 2
            Element 0:
                Numpy Array with shape (28, 28), dtype float64
            Element 1:
                Numpy Array with shape (28, 28), dtype float64
        Element 2:
            int: 42
        Element 3:
            str: Hello World"""

    indent = "  " * level  # Indentation for better readability
    if isinstance(lst, list) or isinstance(lst, tuple):
        print(f"{indent}List of length {len(lst)}")
        for i, item in enumerate(lst):
            print(f"{indent}  Element {i}")
            inspect_list_structure(item, level + 1)  # Recursion for sublists
    elif isinstance(lst, np.ndarray):
        print(f"{indent}Numpy Array with shape {lst.shape}, dtype {lst.dtype}")
    else:
        print(f"{indent}{type(lst).__name__}: {lst}")

