import matplotlib.pyplot as plt
import cv2


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