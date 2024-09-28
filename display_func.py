import matplotlib.pyplot as plt


def show_image(img1, img1_title='Original Image', is_im1_grey=False, img2=None, img2_title='Filtered Image', is_im2_grey=False):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax1.axis('off')
    ax2.axis('off')

    # Display the original and the filtered image
    if is_im1_grey:
        ax1.imshow(
            img1, cmap='gray'), ax1.set_title(img1_title)
    else:
        ax1.imshow(img1), ax1.set_title(img1_title)
    if img2 is not None:
        if is_im2_grey:
            ax2.imshow(img2, cmap='gray'), ax2.set_title(
                img2_title)
        else:
            ax2.imshow(img2), ax2.set_title(img2_title)

    plt.show()
